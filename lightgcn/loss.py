import torch


def bpr_loss(
    user_embs: torch.Tensor,
    pos_item_embs: torch.Tensor,
    neg_item_embs: torch.Tensor,
    user_0embs: torch.Tensor,
    pos_item_0embs: torch.Tensor,
    neg_item_0embs: torch.Tensor,
    reg_param: float,
) -> torch.Tensor:
    regularization = (
        0.5
        * reg_param
        * (user_0embs.norm().pow(2) + pos_item_0embs.norm().pow(2) + neg_item_0embs.norm().pow(2))
    )
    pos_inner_products = torch.mul(user_embs, pos_item_embs)
    neg_inner_products = torch.mul(user_embs, neg_item_embs)

    pos_term = torch.sum(pos_inner_products, dim=1)
    neg_term = torch.sum(neg_inner_products, dim=1)
    # TODO: take log.
    loss = torch.mean(torch.nn.functional.softplus(neg_term - pos_term))
    return loss + regularization
