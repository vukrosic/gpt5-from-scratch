import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import numpy as np
from transformers import AutoTokenizer
from dataclasses import dataclass
from typing import List, Optional, Tuple
import warnings
import os
import pickle
from tqdm import tqdm
import torch.serialization
warnings.filterwarnings('ignore')

# Import the model classes from the training file
import sys
sys.path.append('.')
from train_gpt5 import (
    ModelConfig, MinimalLLM, Rotary, Qwen3Attention, 
    SwiGLUFeedForward, TransformerBlock, repeat_kv
)

def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f" Set all seeds to {seed}")

class TextGenerator:
    """Text generation class for the trained model"""
    
    def __init__(self, model_path: str = "final_model.pt", tokenizer_path: str = "HuggingFaceTB/SmolLM-135M", device: str = "auto"):
        """
        Initialize the text generator
        
        Args:
            model_path: Path to the saved model checkpoint (default: final_model.pt)
            tokenizer_path: Path to the tokenizer (default uses the same as training)
            device: Device to run inference on ("auto", "cpu", "cuda")
        """
        self.device = self._get_device(device)
        print(f"🔧 Using device: {self.device}")
        
        # Load tokenizer
        print("📚 Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, token=False)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        print("🤖 Loading model...")
        self.model, self.config = self._load_model(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ Model loaded successfully!")
        print(f"   Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"   Vocabulary size: {self.config.vocab_size}")
        print(f"   Max sequence length: {self.config.max_seq_len}")
    
    def _get_device(self, device: str) -> torch.device:
        """Determine the best device to use"""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            else:
                return torch.device("cpu")
        else:
            return torch.device(device)
    
    def _load_model(self, model_path: str) -> Tuple[MinimalLLM, ModelConfig]:
        """Load the trained model from checkpoint"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
        
        # Add ModelConfig to safe globals for PyTorch 2.6+ compatibility
        torch.serialization.add_safe_globals([ModelConfig])
        
        try:
            # Try loading with weights_only=True first (PyTorch 2.6+ default)
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        except Exception as e:
            print(f"⚠️  weights_only=True failed, trying weights_only=False: {e}")
            # Fallback to weights_only=False for older checkpoints
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Extract config and create model
        config = checkpoint['config']
        model = MinimalLLM(config)
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model, config
    
    def generate(
        self, 
        prompt: str, 
        max_length: int = 100, 
        temperature: float = 0.8, 
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        num_return_sequences: int = 1,
        stop_tokens: Optional[List[str]] = None
    ) -> List[str]:
        """
        Generate text from a prompt
        
        Args:
            prompt: Input text prompt
            max_length: Maximum length of generated text (including prompt)
            temperature: Sampling temperature (higher = more random)
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            do_sample: Whether to use sampling (False for greedy decoding)
            num_return_sequences: Number of sequences to generate
            stop_tokens: List of tokens to stop generation at
            
        Returns:
            List of generated text sequences
        """
        # Tokenize prompt
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
        input_ids = input_ids.to(self.device)
        
        # Convert stop tokens to IDs
        stop_token_ids = []
        if stop_tokens:
            for token in stop_tokens:
                token_id = self.tokenizer.encode(token, add_special_tokens=False)
                if token_id:
                    stop_token_ids.extend(token_id)
        
        generated_sequences = []
        
        for _ in range(num_return_sequences):
            # Generate sequence
            generated_ids = self._generate_sequence(
                input_ids=input_ids,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                stop_token_ids=stop_token_ids
            )
            
            # Decode to text
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            generated_sequences.append(generated_text)
        
        return generated_sequences
    
    def _generate_sequence(
        self,
        input_ids: torch.Tensor,
        max_length: int,
        temperature: float,
        top_p: float,
        top_k: int,
        do_sample: bool,
        stop_token_ids: List[int]
    ) -> torch.Tensor:
        """Generate a single sequence using the model"""
        
        current_ids = input_ids.clone()
        generated_length = current_ids.shape[1]
        
        with torch.no_grad():
            while generated_length < max_length:
                # Get model predictions
                logits = self.model(current_ids)
                next_token_logits = logits[0, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                    next_token_logits[top_k_indices] = top_k_logits
                
                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                    sorted_indices_to_remove[0] = 0
                    
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample or take argmax
                if do_sample:
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Check for stop tokens
                if next_token.item() in stop_token_ids:
                    break
                
                # Append to sequence
                current_ids = torch.cat([current_ids, next_token], dim=1)
                generated_length += 1
        
        return current_ids[0]
    
    def get_perplexity(self, text: str) -> float:
        """Calculate perplexity of the given text"""
        # Tokenize text
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        
        if len(tokens) < 2:
            return float('inf')
        
        # Create sequences for evaluation
        sequences = []
        for i in range(len(tokens) - 1):
            sequences.append((tokens[i], tokens[i + 1]))
        
        total_loss = 0
        total_tokens = 0
        
        self.model.eval()
        with torch.no_grad():
            for input_token, target_token in sequences:
                # Create input tensor
                input_tensor = torch.tensor([[input_token]], device=self.device)
                target_tensor = torch.tensor([[target_token]], device=self.device)
                
                # Get model prediction
                logits = self.model(input_tensor)
                loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), target_tensor.view(-1))
                
                total_loss += loss.item()
                total_tokens += 1
        
        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)
        
        return perplexity

def interactive_mode(generator: TextGenerator):
    """Run interactive text generation mode"""
    print("\n🎭 Interactive Text Generation Mode")
    print("Type 'quit' to exit, 'help' for commands")
    print("=" * 50)
    
    while True:
        try:
            prompt = input("\n Enter your prompt: ").strip()
            
            if prompt.lower() == 'quit':
                print("👋 Goodbye!")
                break
            elif prompt.lower() == 'help':
                print("\n Available commands:")
                print("  help - Show this help message")
                print("  quit - Exit the program")
                print("  settings - Show current generation settings")
                print("  sample - Generate with sampling")
                print("  greedy - Generate with greedy decoding")
                print("\n💡 Tips:")
                print("  - Use 'sample' or 'greedy' prefix to change generation mode")
                print("  - Example: 'sample The quick brown fox'")
                continue
            elif prompt.lower() == 'settings':
                print("\n⚙️ Current settings:")
                print("  Temperature: 0.8")
                print("  Top-p: 0.9")
                print("  Top-k: 50")
                print("  Max length: 100")
                continue
            
            # Check for mode prefixes
            do_sample = True
            if prompt.startswith('sample '):
                prompt = prompt[7:]
                do_sample = True
            elif prompt.startswith('greedy '):
                prompt = prompt[7:]
                do_sample = False
            
            if not prompt:
                continue
            
            print(f"\n Generating text...")
            generated_texts = generator.generate(
                prompt=prompt,
                max_length=100,
                temperature=0.8,
                top_p=0.9,
                top_k=50,
                do_sample=do_sample,
                num_return_sequences=1
            )
            
            print(f"\n✨ Generated text:")
            print("-" * 40)
            for i, text in enumerate(generated_texts, 1):
                print(f"{i}. {text}")
            print("-" * 40)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    # Set seed
    set_seed(42)
    
    # Initialize generator with default settings
    try:
        generator = TextGenerator("final_model.pt")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Run interactive mode
    interactive_mode(generator)

if __name__ == "__main__":
    main() 