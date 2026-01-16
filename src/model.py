"""
VLM-VLA (视觉语言模型 - 视觉语言动作) Agent 模型
用于多模态变化检测与动作预测的架构
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import (
    CLIPModel,
    CLIPProcessor,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from .config import Config


class VLM_ActionAgent(nn.Module):
    """
    VLM-VLA（视觉语言模型-视觉语言动作）Agent

    核心架构说明：
        1. 视觉编码器 (CLIP-ViT-B32)：
           - 冻结的参数（不训练）
           - 将两张时序影像编码为视觉特征向量
           - 输出维度：512 维

        2. 特征投影层：
           - 将双时相特征（512*2=1024维）投影到 LLM 嵌入空间（1024维）
           - 实现维度适配

        3. 语言模型 (Qwen2.5-0.5B with LoRA)：
           - 使用 LoRA 进行参数高效微调（只训练 ~0.1% 参数）
           - 4位量化减少显存占用
           - 进行文本理解和语义编码

        4. 特征融合模块：
           - 组合视觉特征和语言特征
           - 产生多模态表示

        5. 动作预测头 (MLP)：
           - 3层 MLP 网络
           - 预测动作向量：[中心x, 中心y, 尺度]
           - 输出范围：[0, 1]³（归一化）

    数据流：
        输入: (batch_size, 3, 224, 224) 图像对 + 文本
             ↓
        CLIP编码：2 × (B, 512)
             ↓
        拼接：(B, 1024)
             ↓
        投影：(B, 1024)
             ↓
        LLM编码 + 融合：(B, 1024)
             ↓
        动作预测：(B, 3)
    """

    def __init__(
        self,
        clip_path: str = None,
        llm_path: str = None,
        freeze_vision: bool = True,
        use_lora: bool = True,
        use_4bit: bool = True,
    ):
        """
        初始化 VLM-VLA Agent

        参数初始化流程：
        1. 加载冻结的 CLIP 视觉编码器
        2. 加载 LLM 并应用 LoRA
        3. 初始化投影层和动作头

        Args:
            clip_path (str): 本地 CLIP 模型路径（预训练权重）
            llm_path (str): 本地 LLM 模型路径（预训练权重）
            freeze_vision (bool): 是否冻结 CLIP 视觉编码器（推荐 True）
            use_lora (bool): 是否对 LLM 使用 LoRA 微调（推荐 True）
            use_4bit (bool): 是否对 LLM 使用 4 位量化（节省显存）
        """
        super().__init__()

        self.clip_path = clip_path or Config.CLIP_PATH
        self.llm_path = llm_path or Config.LLM_PATH
        self.freeze_vision = freeze_vision
        self.use_lora = use_lora

        print(f"\n{'='*60}")
        print("Initializing VLM-VLA Agent")
        print(f"{'='*60}")

        # Load vision encoder (CLIP)
        print(f"正在从以下路径加载模型: {self.llm_path}")
        self._load_vision_encoder()

        # Load LLM
        self._load_llm(use_4bit=use_4bit)

        # Create projector: Concatenated temporal CLIP features -> LLM embedding dim
        # Input: 2 * VISION_OUTPUT_DIM (双时相图像特征拼接) = 1536
        # Output: LLM_HIDDEN_DIM = 1024
        # 这个投影层将两张图像的特征从 1536 维投影到 1024 维
        self.projector = nn.Linear(
            Config.VISION_OUTPUT_DIM * 2,  # 768 * 2 = 1536
            Config.LLM_HIDDEN_DIM  # 1024
        )

        print(f"✅ Projector 配置:")
        print(f"   输入维度: {Config.VISION_OUTPUT_DIM * 2}")
        print(f"   输出维度: {Config.LLM_HIDDEN_DIM}")

        # Action head: Predict action vector [cx, cy, scale]
        self.action_head = nn.Sequential(
            nn.Linear(Config.LLM_HIDDEN_DIM, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 3),  # [cx, cy, scale]
            nn.Sigmoid(),  # Normalize to [0, 1]
        )

        # Loss functions
        self.text_loss_fn = nn.CrossEntropyLoss()
        self.action_loss_fn = nn.MSELoss()

        print(f"{'='*60}\n")

    def _load_vision_encoder(self):
        """Load CLIP vision encoder from local path"""
        print(f"Loading CLIP model from: {self.clip_path}")

        try:
            self.clip_model = CLIPModel.from_pretrained(
                self.clip_path,
                local_files_only=True,
            )
            self.clip_processor = CLIPProcessor.from_pretrained(
                self.clip_path,
                local_files_only=True,
            )

            # Freeze vision encoder
            if self.freeze_vision:
                for param in self.clip_model.vision_model.parameters():
                    param.requires_grad = False
                print("✅ CLIP vision encoder loaded and frozen")
            else:
                print("✅ CLIP vision encoder loaded (trainable)")

        except Exception as e:
            print(f"❌ Error loading CLIP model: {e}")
            raise

    def _load_llm(self, use_4bit: bool = True):
        """Load LLM from local path with optional 4-bit quantization"""
        print(f"Loading LLM model from: {self.llm_path}")

        try:
            # Configure 4-bit quantization if requested
            if use_4bit:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
                print("✅ Using 4-bit quantization")
            else:
                bnb_config = None
                print("⚠️  Not using quantization (higher memory usage)")

            # Load model
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                self.llm_path,
                local_files_only=True,
                device_map="auto",
                quantization_config=bnb_config if use_4bit else None,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )

            # Load tokenizer
            self.llm_tokenizer = AutoTokenizer.from_pretrained(
                self.llm_path,
                local_files_only=True,
                trust_remote_code=True,
            )

            print("✅ LLM model loaded successfully")

            # Apply LoRA if requested
            if self.use_lora:
                self._apply_lora()

        except Exception as e:
            print(f"❌ Error loading LLM model: {e}")
            raise

    def _apply_lora(self):
        """Apply LoRA to LLM for efficient fine-tuning"""
        print("Applying LoRA to LLM...")

        lora_config = LoraConfig(
            r=Config.LORA_R,
            lora_alpha=Config.LORA_ALPHA,
            target_modules=Config.LORA_TARGET_MODULES,
            lora_dropout=Config.LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM",
        )

        self.llm_model = get_peft_model(self.llm_model, lora_config)

        # Print trainable parameters
        trainable_params = sum(p.numel() for p in self.llm_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.llm_model.parameters())

        print(f"✅ LoRA applied")
        print(f"   Trainable params: {trainable_params:,}")
        print(f"   Total params: {total_params:,}")
        print(f"   Trainable ratio: {100 * trainable_params / total_params:.2f}%")

    def extract_vision_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract features from images using CLIP

        Args:
            images: Tensor of shape (B, 3, H, W) - Already preprocessed

        Returns:
            Tensor of shape (B, vision_hidden_dim) - CLIP features
        """
        with torch.no_grad():  # Don't track gradients for frozen encoder
            clip_outputs = self.clip_model.vision_model(images)
            # Use pooled output (CLS token)
            pooled_output = clip_outputs.pooler_output

        return pooled_output

    def forward(
        self,
        images_t1: torch.Tensor,
        images_t2: torch.Tensor,
        captions: list,
        action_targets: Optional[torch.Tensor] = None,
        bbox_targets: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for VLM-VLA Agent

        Args:
            images_t1: Temporal image 1, shape (B, 3, H, W)
            images_t2: Temporal image 2, shape (B, 3, H, W)
            captions: List of caption strings, length B
            action_targets: Ground truth action vectors, shape (B, 3) if provided
            bbox_targets: Ground truth bboxes, shape (B, 4) if provided

        Returns:
            Dictionary containing:
                - 'action_pred': Predicted action vectors (B, 3)
                - 'text_loss': Loss for text understanding (scalar) if targets provided
                - 'action_loss': Loss for action prediction (scalar) if targets provided
                - 'total_loss': Total loss (scalar) if targets provided
                - 'vision_features_t1': Vision features from image 1
                - 'vision_features_t2': Vision features from image 2
        """
        batch_size = images_t1.shape[0]

        # Extract vision features from both temporal images
        vision_features_t1 = self.extract_vision_features(images_t1)  # (B, 768)
        vision_features_t2 = self.extract_vision_features(images_t2)  # (B, 768)

        # Concatenate temporal features and project to LLM space
        temporal_features = torch.cat([vision_features_t1, vision_features_t2], dim=-1)  # (B, 1536)

        # Project to LLM hidden dim
        # temporal_features: (B, 1536) -> (B, 1024)
        temporal_features = self.projector(temporal_features)  # (B, 1024)

        # Tokenize captions
        caption_tokens = self.llm_tokenizer(
            captions,
            padding=True,
            truncation=True,
            max_length=Config.MAX_TEXT_LENGTH,
            return_tensors="pt",
        ).to(images_t1.device)

        # Forward through LLM
        # Combine vision features with text embeddings
        llm_outputs = self.llm_model(
            input_ids=caption_tokens['input_ids'],
            attention_mask=caption_tokens['attention_mask'],
            output_hidden_states=True,
        )

        # Get final hidden state and combine with vision features
        llm_hidden_states = llm_outputs.hidden_states[-1]  # (B, seq_len, hidden_dim)
        llm_pooled = llm_hidden_states[:, 0, :]  # Use CLS token equivalent (B, hidden_dim)

        # Fuse vision and language features
        fused_features = temporal_features + llm_pooled  # Element-wise addition (B, hidden_dim)

        # Predict action vector
        action_pred = self.action_head(fused_features)  # (B, 3)

        outputs = {
            'action_pred': action_pred,
            'vision_features_t1': vision_features_t1,
            'vision_features_t2': vision_features_t2,
            'fused_features': fused_features,
        }

        # Calculate losses if targets provided
        if action_targets is not None:
            action_loss = self.action_loss_fn(action_pred, action_targets)
            outputs['action_loss'] = action_loss
            outputs['total_loss'] = action_loss

        return outputs

    def get_trainable_params(self):
        """Get all trainable parameters"""
        return [p for p in self.parameters() if p.requires_grad]

    def get_model_size(self) -> Dict[str, int]:
        """Get model size statistics"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'frozen_params': total_params - trainable_params,
        }


def create_model(
    clip_path: str = None,
    llm_path: str = None,
    use_lora: bool = True,
    use_4bit: bool = True,
) -> VLM_ActionAgent:
    """
    Factory function to create VLM-VLA Agent

    Args:
        clip_path: Path to CLIP model
        llm_path: Path to LLM model
        use_lora: Whether to use LoRA
        use_4bit: Whether to use 4-bit quantization

    Returns:
        VLM_ActionAgent model
    """
    model = VLM_ActionAgent(
        clip_path=clip_path,
        llm_path=llm_path,
        freeze_vision=True,
        use_lora=use_lora,
        use_4bit=use_4bit,
    )

    # Print model size
    model_size = model.get_model_size()
    print(f"\nModel size:")
    print(f"  Total parameters: {model_size['total_params']:,}")
    print(f"  Trainable parameters: {model_size['trainable_params']:,}")
    print(f"  Frozen parameters: {model_size['frozen_params']:,}")

    return model


if __name__ == "__main__":
    # Test model creation
    print("Testing model creation...")
    model = create_model()

    # Test forward pass
    batch_size = 2
    images_t1 = torch.randn(batch_size, 3, 224, 224)
    images_t2 = torch.randn(batch_size, 3, 224, 224)
    captions = ["significant change in building area", "no change detected"]
    action_targets = torch.rand(batch_size, 3)

    outputs = model(images_t1, images_t2, captions, action_targets)

    print(f"\nForward pass successful!")
    print(f"  action_pred shape: {outputs['action_pred'].shape}")
    print(f"  total_loss: {outputs['total_loss'].item():.4f}")

