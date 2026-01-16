"""
Training script for VLM-VLA Agent
Handles the main training loop with validation, checkpointing, and logging
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from .config import Config
from .dataset import create_dataloaders
from .model import create_model


class Trainer:
    """
    VLM-VLA Agent 训练器

    主要职责：
        1. 数据加载：从多种格式加载数据集
        2. 模型初始化：创建和配置模型
        3. 训练循环：执行 epoch 训练和验证
        4. 检查点管理：保存最优模型
        5. 指标记录：跟踪训练进度

    特性：
        - 自动错误恢复（跳过损坏的批次）
        - 混合精度训练支持
        - 梯度累积支持
        - 学习率调度支持
        - 详细的进度显示
    """

    def __init__(self, config: Config = None):
        """
        初始化训练器

        初始化步骤：
        1. 检测设备（GPU/CPU）
        2. 创建输出目录
        3. 初始化指标跟踪

        Args:
            config (Config): 配置对象（使用 Config 类的默认值）
        """
        self.config = config or Config  # 使用传入的配置或默认配置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检测可用设备

        print(f"\n{'='*60}")
        print("Initializing Trainer")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
            print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

        # Create output directory
        os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)

        # Initialize training state
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')

        # Create timestamp for checkpoints
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.checkpoint_dir = Path(self.config.OUTPUT_DIR) / f"checkpoint_{self.timestamp}"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Logging
        self.metrics = {
            'train': {
                'loss': [],
                'action_loss': [],
                'learning_rate': [],
            },
            'val': {
                'loss': [],
                'action_loss': [],
            }
        }

        print(f"Checkpoint directory: {self.checkpoint_dir}")
        print(f"{'='*60}\n")

    def load_data(self):
        """Load training and validation dataloaders"""
        print(f"{'='*60}")
        print("Loading Datasets")
        print(f"{'='*60}")

        self.train_loader, self.val_loader = create_dataloaders(
            batch_size=self.config.BATCH_SIZE,
            num_workers=self.config.NUM_WORKERS,
            test_split=0.1,
            seed=self.config.SEED,
        )

        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
        print(f"{'='*60}\n")

    def setup_model(self):
        """Initialize model and optimizer"""
        print(f"正在从以下路径加载模型: {self.config.LLM_PATH}")

        self.model = create_model(
            clip_path=self.config.CLIP_PATH,
            llm_path=self.config.LLM_PATH,
            use_lora=True,
            use_4bit=self.config.USE_4BIT,
        )

        self.model.to(self.device)

        # Setup optimizer - only optimize trainable parameters
        trainable_params = self.model.get_trainable_params()

        self.optimizer = optim.AdamW(
            trainable_params,
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY,
            betas=(0.9, 0.95),
        )

        # Setup learning rate scheduler
        total_steps = len(self.train_loader) * self.config.MAX_EPOCHS
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=self.config.LEARNING_RATE * 0.1
        )

        # Mixed precision training
        self.scaler = GradScaler() if self.config.USE_MIXED_PRECISION else None

        print(f"\n{'='*60}")
        print("Optimizer Configuration")
        print(f"{'='*60}")
        print(f"Learning rate: {self.config.LEARNING_RATE}")
        print(f"Weight decay: {self.config.WEIGHT_DECAY}")
        print(f"Mixed precision: {self.config.USE_MIXED_PRECISION}")
        print(f"Gradient accumulation steps: {self.config.GRADIENT_ACCUMULATION_STEPS}")
        print(f"Total training steps: {total_steps}")
        print(f"{'='*60}\n")

    def train_epoch(self):
        """
        执行一个 epoch 的训练

        训练流程：
        1. 遍历训练批次
        2. 前向传播计算损失
        3. 反向传播更新梯度
        4. 梯度累积处理
        5. 优化器步骤和学习率更新

        特点：
        - 支持混合精度训练加速
        - 支持梯度累积增加有效批大小
        - 自动错误处理和批次跳过
        - 定期保存检查点

        Returns:
            tuple: (平均训练损失, 平均动作损失)
        """
        self.model.train()  # 设置模型为训练模式

        total_loss = 0.0  # 累积损失
        total_action_loss = 0.0  # 累积动作损失
        skipped_batches = 0  # 跳过的批次计数（错误恢复）

        pbar = tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            desc=f"Epoch {self.epoch + 1}/{self.config.MAX_EPOCHS}",
            leave=True,
        )

        for batch_idx, batch in pbar:
            try:
                # 将批次数据移到设备（GPU/CPU）
                images_t1 = batch['image_t1'].to(self.device)  # 时序影像 1
                images_t2 = batch['image_t2'].to(self.device)  # 时序影像 2
                action_targets = batch['action_vector'].to(self.device)  # 目标动作向量
                captions = batch['caption']  # 变化描述文本

                # 前向传播（支持混合精度加速）
                if self.config.USE_MIXED_PRECISION and self.scaler:
                    with autocast(dtype=torch.bfloat16):
                        outputs = self.model(
                            images_t1,
                            images_t2,
                            captions,
                            action_targets=action_targets,
                        )
                        loss = outputs['total_loss']

                        # Normalize loss by gradient accumulation steps
                        loss = loss / self.config.GRADIENT_ACCUMULATION_STEPS

                    # Backward pass with gradient scaling
                    self.scaler.scale(loss).backward()
                else:
                    outputs = self.model(
                        images_t1,
                        images_t2,
                        captions,
                        action_targets=action_targets,
                    )
                    loss = outputs['total_loss']
                    loss = loss / self.config.GRADIENT_ACCUMULATION_STEPS
                    loss.backward()

                # Gradient accumulation
                if (batch_idx + 1) % self.config.GRADIENT_ACCUMULATION_STEPS == 0:
                    # Gradient clipping
                    if self.config.USE_MIXED_PRECISION and self.scaler:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.get_trainable_params(),
                            self.config.MAX_GRAD_NORM
                        )
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.get_trainable_params(),
                            self.config.MAX_GRAD_NORM
                        )
                        self.optimizer.step()

                    self.optimizer.zero_grad()
                    self.scheduler.step()
                    self.global_step += 1

            except Exception as e:
                print(f"\n⚠️  批处理 {batch_idx} 出现错误: {e}")
                import traceback
                traceback.print_exc()
                skipped_batches += 1
                self.optimizer.zero_grad()  # 清空梯度
                continue

            # Accumulate loss (multiply back by accumulation steps for reporting)
            try:
                total_loss += loss.item() * self.config.GRADIENT_ACCUMULATION_STEPS

                if 'action_loss' in outputs:
                    action_loss = outputs['action_loss'].item()
                    total_action_loss += action_loss * self.config.GRADIENT_ACCUMULATION_STEPS

                # Update progress bar
                batch_count = batch_idx + 1 - skipped_batches
                if batch_count > 0:
                    avg_loss = total_loss / batch_count
                else:
                    avg_loss = 0.0

                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}',
                    'skipped': skipped_batches,
                })
            except Exception as e:
                print(f"\n⚠️  损失计算出错: {e}")

            # Periodic checkpoint and validation
            if self.global_step % self.config.SAVE_INTERVAL == 0 and self.global_step > 0:
                print(f"\n📊 Global step {self.global_step} - Saving checkpoint...")
                self.save_checkpoint(step=self.global_step)

            if self.config.DEBUG and batch_idx >= 5:
                print("🐛 Debug mode: Stopping after 5 batches")
                break

        # Average loss for epoch
        avg_epoch_loss = total_loss / len(self.train_loader)
        avg_action_loss = total_action_loss / len(self.train_loader) if total_action_loss > 0 else 0.0

        self.metrics['train']['loss'].append(avg_epoch_loss)
        self.metrics['train']['action_loss'].append(avg_action_loss)
        self.metrics['train']['learning_rate'].append(self.optimizer.param_groups[0]['lr'])

        return avg_epoch_loss, avg_action_loss

    def validate(self):
        """Validate model on validation set"""
        self.model.eval()

        total_loss = 0.0
        total_action_loss = 0.0
        skipped_batches = 0

        with torch.no_grad():
            pbar = tqdm(
                enumerate(self.val_loader),
                total=len(self.val_loader),
                desc="Validating",
                leave=True,
            )

            for batch_idx, batch in pbar:
                try:
                    images_t1 = batch['image_t1'].to(self.device)
                    images_t2 = batch['image_t2'].to(self.device)
                    action_targets = batch['action_vector'].to(self.device)
                    captions = batch['caption']

                    if self.config.USE_MIXED_PRECISION:
                        with autocast(dtype=torch.bfloat16):
                            outputs = self.model(
                                images_t1,
                                images_t2,
                                captions,
                                action_targets=action_targets,
                            )
                    else:
                        outputs = self.model(
                            images_t1,
                            images_t2,
                            captions,
                            action_targets=action_targets,
                        )

                    loss = outputs['total_loss'].item()
                    total_loss += loss

                    if 'action_loss' in outputs:
                        total_action_loss += outputs['action_loss'].item()

                    batch_count = batch_idx + 1 - skipped_batches
                    pbar.set_postfix({'loss': f'{loss:.4f}', 'skipped': skipped_batches})

                except Exception as e:
                    print(f"\n⚠️  验证批处理 {batch_idx} 出现错误: {e}")
                    skipped_batches += 1
                    continue

        # 计算平均验证损失，排除跳过的批次
        valid_batches = len(self.val_loader) - skipped_batches
        if valid_batches > 0:
            avg_val_loss = total_loss / valid_batches
            avg_val_action_loss = total_action_loss / valid_batches if total_action_loss > 0 else 0.0
        else:
            avg_val_loss = float('inf')
            avg_val_action_loss = 0.0
            print(f"⚠️  所有验证批处理都被跳过！")

        self.metrics['val']['loss'].append(avg_val_loss)
        self.metrics['val']['action_loss'].append(avg_val_action_loss)

        return avg_val_loss, avg_val_action_loss

    def save_checkpoint(self, step: int = None, is_best: bool = False):
        """
        Save model checkpoint

        优化：只保存模型权重和训练指标，不保存优化器状态以节省磁盘空间
        这使得检查点大小减少约 70%，适应 Kaggle 的磁盘限制

        Args:
            step: Global training step
            is_best: Whether this is the best model
        """
        checkpoint_name = f"checkpoint_step_{step}.pt" if step else "checkpoint_latest.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_name

        # 轻量级检查点：只保存必要的信息以恢复模型推理
        # 为了节省 Kaggle 磁盘空间，移除了优化器和调度器的完整状态
        # 这允许继续训练需要重新创建优化器，但节省大量磁盘空间
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            # 注：optimizer_state_dict 和 scheduler_state_dict 已移除以节省磁盘空间
            'metrics': self.metrics,
            'config': {
                'max_epochs': self.config.MAX_EPOCHS,
                'batch_size': self.config.BATCH_SIZE,
                'learning_rate': self.config.LEARNING_RATE,
            }
        }

        try:
            torch.save(checkpoint, checkpoint_path)
            print(f"✅ Checkpoint saved: {checkpoint_path}")
        except RuntimeError as e:
            print(f"⚠️  检查点保存失败: {e}")
            print("⚠️  可能是磁盘空间不足，继续训练...")
            return

        # Also save best checkpoint
        if is_best or self.best_loss > (self.metrics['val']['loss'][-1] if self.metrics['val']['loss'] else float('inf')):
            best_path = self.checkpoint_dir / "checkpoint_best.pt"
            try:
                torch.save(checkpoint, best_path)
                self.best_loss = self.metrics['val']['loss'][-1] if self.metrics['val']['loss'] else float('inf')
                print(f"✅ Best checkpoint updated: {best_path}")
            except RuntimeError as e:
                print(f"⚠️  最优检查点保存失败: {e}")

    def cleanup_old_checkpoints(self, keep_count: int = 3):
        """
        清理旧检查点以节省磁盘空间
        只保留最新的 keep_count 个检查点

        Args:
            keep_count: 要保留的最新检查点数量（默认 3）
        """
        checkpoint_files = sorted(
            self.checkpoint_dir.glob("checkpoint_step_*.pt"),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )

        # 保留最新的 keep_count 个检查点，删除其他的
        if len(checkpoint_files) > keep_count:
            for old_checkpoint in checkpoint_files[keep_count:]:
                try:
                    old_checkpoint.unlink()
                    print(f"🗑️  删除旧检查点: {old_checkpoint.name}")
                except Exception as e:
                    print(f"⚠️  无法删除 {old_checkpoint.name}: {e}")

    def save_metrics(self):
        """Save training metrics to JSON"""
        metrics_path = self.checkpoint_dir / "metrics.json"

        # Convert lists to summary stats for readability
        metrics_summary = {
            'train': {
                'final_loss': self.metrics['train']['loss'][-1] if self.metrics['train']['loss'] else None,
                'min_loss': min(self.metrics['train']['loss']) if self.metrics['train']['loss'] else None,
                'max_loss': max(self.metrics['train']['loss']) if self.metrics['train']['loss'] else None,
            },
            'val': {
                'final_loss': self.metrics['val']['loss'][-1] if self.metrics['val']['loss'] else None,
                'min_loss': min(self.metrics['val']['loss']) if self.metrics['val']['loss'] else None,
                'max_loss': max(self.metrics['val']['loss']) if self.metrics['val']['loss'] else None,
            }
        }

        with open(metrics_path, 'w') as f:
            json.dump(metrics_summary, f, indent=2)

        print(f"✅ Metrics saved: {metrics_path}")

    def train(self):
        """Main training loop"""
        print(f"\n{'='*60}")
        print("Starting Training")
        print(f"{'='*60}\n")

        # Load data
        self.load_data()

        # Setup model
        self.setup_model()

        # Training loop
        start_time = time.time()

        for epoch in range(self.config.MAX_EPOCHS):
            self.epoch = epoch

            # Train epoch
            train_loss, train_action_loss = self.train_epoch()
            print(f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f}, Action Loss: {train_action_loss:.4f}")

            # Validate
            val_loss, val_action_loss = self.validate()
            print(f"Epoch {epoch + 1} - Val Loss: {val_loss:.4f}, Action Loss: {val_action_loss:.4f}")

            # Save checkpoint
            is_best = val_loss < self.best_loss
            self.save_checkpoint(step=self.epoch, is_best=is_best)

            # Cleanup old checkpoints to save disk space
            self.cleanup_old_checkpoints(keep_count=3)

            print(f"{'='*60}")

        # Final checkpoint
        self.save_checkpoint(step=self.global_step)
        self.save_metrics()

        elapsed_time = time.time() - start_time
        print(f"\n{'='*60}")
        print("Training Completed")
        print(f"{'='*60}")
        print(f"Total time: {elapsed_time / 3600:.2f} hours")
        print(f"Checkpoints saved in: {self.checkpoint_dir}")
        print(f"{'='*60}\n")


def main():
    """Main entry point"""
    # Print configuration
    Config.print_config()
    Config.verify_paths()

    # Create trainer and start training
    trainer = Trainer(config=Config)
    trainer.train()


if __name__ == "__main__":
    main()

