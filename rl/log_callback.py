from transformers import TrainerCallback
import os
class CustomLogCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if args.local_rank != 0:
            return
        # 手动记录自定义指标
        log_dir = args.logging_dir
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, 'log.txt')
        # 写入关键训练指标
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"[Step {state.global_step}]\n")
            f.write(f"Epoch: {state.epoch:.2f}\n")
            f.write(f"Loss: {logs.get('loss', 0.0):.4f}\n")
            f.write(f"Reward: {logs.get('reward', 0.0):.4f}\n")
            f.write(f"Reward Std: {logs.get('reward_std', 0.0):.4f}\n")
            f.write(f"Learning Rate: {logs.get('learning_rate', 0.0):.8f}\n")
            f.write(f"Step Time: {state.log_history[-1].get('step_time', 0.0):.6f}s\n\n")
