import torch
import torch.nn.functional as F
from conditional_flow_matching import ConditionalFlowMatcher

def compute_smoothness_loss(trajectory, action_mask=None):
    """
    计算平滑度损失 (基于jerk的有限差分近似)
    L_smooth = sum ||x_{k+3} - 3*x_{k+2} + 3*x_{k+1} - x_k||^2
    
    Args:
        trajectory: shape (batch_size, seq_len, action_dim)
        action_mask: optional mask for valid actions
    """
    batch_size, seq_len, action_dim = trajectory.shape
    
    if seq_len < 4:
        return torch.tensor(0.0, device=trajectory.device)
    
    smooth_loss = 0.0
    for k in range(seq_len - 3):
        jerk = (trajectory[:, k+3] - 3*trajectory[:, k+2] + 
                3*trajectory[:, k+1] - trajectory[:, k])
        loss_k = torch.sum(jerk ** 2, dim=-1)  # L2 norm squared
        
        if action_mask is not None:
            # Apply mask to only consider valid timesteps
            mask_k = action_mask[:, k] * action_mask[:, k+1] * \
                     action_mask[:, k+2] * action_mask[:, k+3]
            loss_k = loss_k * mask_k
        
        smooth_loss += loss_k.mean()
    
    return smooth_loss

def compute_safety_loss(trajectory, success_field_fn, epsilon=0.1, action_mask=None):
    """
    计算安全性损失 (基于对数障碍函数)
    L_safe = -sum log(max(d_k - epsilon, 0))
    
    Args:
        trajectory: shape (batch_size, seq_len, action_dim)
        success_field_fn: 函数，输入waypoint返回success field值
        epsilon: 安全边界阈值
        action_mask: optional mask for valid actions
    """
    batch_size, seq_len, action_dim = trajectory.shape
    
    safe_loss = 0.0
    for k in range(seq_len):
        # 获取每个waypoint的success field值 (作为到障碍物距离的代理)
        waypoint = trajectory[:, k]  # shape: (batch_size, action_dim)
        
        # 计算到障碍物的距离 (这里假设success_field_fn返回值与距离成正比)
        # 实际实现中需要根据你的success field定义来调整
        d_k = success_field_fn(waypoint)  # shape: (batch_size,)
        
        # 计算log barrier
        safe_distance = torch.clamp(d_k - epsilon, min=1e-8)  # 避免log(0)
        loss_k = -torch.log(safe_distance)
        
        if action_mask is not None:
            loss_k = loss_k * action_mask[:, k]
        
        safe_loss += loss_k.mean()
    
    return safe_loss

# 主训练循环
def train_regularized_cfm(model, noise, naction, obsgoal_cond, action_mask, 
                          success_field_fn, lambda_smooth=0.1, lambda_safe=0.01,
                          epsilon=0.1):
    """
    带正则化的CFM训练
    
    Args:
        model: 神经网络模型
        noise: 初始噪声轨迹 x0
        naction: 目标专家轨迹 x1
        obsgoal_cond: 条件信息 z_c
        action_mask: 动作掩码
        success_field_fn: success field函数
        lambda_smooth: 平滑度损失权重
        lambda_safe: 安全性损失权重
        epsilon: 安全边界阈值
    """
    # Flow Matching部分
    FM = ConditionalFlowMatcher(sigma=0.0)
    t, xt, ut = FM.sample_location_and_conditional_flow(x0=noise, x1=naction)
    
    # 预测向量场
    vt = model(
        "noise_pred_net", 
        sample=xt, 
        timestep=t, 
        global_cond=obsgoal_cond
    )
    
    # 基础Flow Matching损失 (L_FM)
    flow_loss = action_reduce(F.mse_loss(vt, ut, reduction="none"), action_mask)
    
    # 计算当前轨迹 (用于正则化)
    # 注意：xt是插值后的轨迹，shape应该是 (batch_size, seq_len * action_dim)
    # 需要reshape成 (batch_size, seq_len, action_dim)
    batch_size = xt.shape[0]
    seq_len = naction.shape[1] if len(naction.shape) > 2 else naction.shape[0] // action_dim
    action_dim = obsgoal_cond.shape[-1] if hasattr(obsgoal_cond, 'shape') else 2  # 假设2D
    
    current_trajectory = xt.view(batch_size, seq_len, action_dim)
    
    # 平滑度正则化 (L_smooth)
    smooth_loss = compute_smoothness_loss(current_trajectory, action_mask)
    
    # 安全性正则化 (L_safe)
    safe_loss = compute_safety_loss(
        current_trajectory, 
        success_field_fn, 
        epsilon=epsilon,
        action_mask=action_mask
    )
    
    # 总损失 (L_total)
    total_loss = flow_loss + lambda_smooth * smooth_loss + lambda_safe * safe_loss
    
    # 返回损失字典，便于监控
    loss_dict = {
        'total_loss': total_loss,
        'flow_loss': flow_loss,
        'smooth_loss': smooth_loss,
        'safe_loss': safe_loss
    }
    
    return loss_dict

# 如果需要单独的action_reduce函数
def action_reduce(loss, action_mask=None):
    """
    根据action mask减少损失
    """
    if action_mask is None:
        return loss.mean()
    
    # Apply mask and compute mean over valid actions
    masked_loss = loss * action_mask.unsqueeze(-1) if len(loss.shape) > len(action_mask.shape) else loss * action_mask
    return masked_loss.sum() / (action_mask.sum() + 1e-8)

# 示例：定义success field函数
def example_success_field_fn(waypoint):
    """
    示例success field函数
    实际实现需要根据你的环境和障碍物信息来定义
    
    Args:
        waypoint: shape (batch_size, action_dim)
    Returns:
        distances: shape (batch_size,)
    """
    # 这里只是一个示例，实际需要根据你的场景实现
    # 比如可以使用预训练的success field网络或者基于地图的距离计算
    
    # 假设有一个预定义的障碍物中心
    obstacle_center = torch.tensor([5.0, 5.0], device=waypoint.device)
    
    # 计算到障碍物中心的距离
    distances = torch.norm(waypoint - obstacle_center, dim=-1)
    
    # 也可以使用神经网络来预测success field
    # distances = success_field_network(waypoint)
    
    return distances

# 使用示例
if __name__ == "__main__":
    # 假设的输入维度
    batch_size = 32
    seq_len = 50
    action_dim = 2
    
    # 创建示例数据
    noise = torch.randn(batch_size, seq_len, action_dim)
    naction = torch.randn(batch_size, seq_len, action_dim)
    obsgoal_cond = torch.randn(batch_size, 256)  # 假设条件维度为256
    action_mask = torch.ones(batch_size, seq_len)  # 全部有效
    
    # 训练
    loss_dict = train_regularized_cfm(
        model=None,  # 需要替换为实际模型
        noise=noise,
        naction=naction,
        obsgoal_cond=obsgoal_cond,
        action_mask=action_mask,
        success_field_fn=example_success_field_fn,
        lambda_smooth=0.1,
        lambda_safe=0.01,
        epsilon=0.1
    )
    
    print(f"Total Loss: {loss_dict['total_loss']}")
    print(f"Flow Loss: {loss_dict['flow_loss']}")
    print(f"Smooth Loss: {loss_dict['smooth_loss']}")
    print(f"Safe Loss: {loss_dict['safe_loss']}")