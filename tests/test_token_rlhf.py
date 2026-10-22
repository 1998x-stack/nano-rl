"""Advanced RLHF causal generation and sequence-level reward tests."""
import importlib.util
from pathlib import Path
import torch

PATH=Path(__file__).resolve().parents[1] / "15-rlhf-from-scratch" / "token_rlhf.py"
spec=importlib.util.spec_from_file_location("nano_token_rlhf",PATH)
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_two_token_prompt_grammar_and_gradients():
    model=mod.MiniGPT().eval()
    prompt, answer=mod.questions(4)
    assert prompt.shape==(4,5) and answer.shape==(4,2)
    lp, values=model.evaluate(prompt,answer)
    assert lp.shape==values.shape==(4,2)
    assert torch.isfinite(lp).all()
    (-lp.mean()).backward()
    assert model.tok.weight.grad is not None


def test_first_token_prediction_cannot_see_second_answer():
    model=mod.MiniGPT().eval()
    prompt,answer=mod.questions(2)
    changed=answer.clone()
    changed[:,1]=(changed[:,1]+1)%10
    with torch.no_grad():
        first=model.evaluate(prompt,answer)[0][:,0]
        second=model.evaluate(prompt,changed)[0][:,0]
    assert torch.allclose(first,second,atol=1e-7)
    generated=model.generate(prompt)
    assert generated.shape==(2,2)
    assert ((generated[:,0]>=0)&(generated[:,0]<=1)).all()
    assert ((generated[:,1]>=0)&(generated[:,1]<=9)).all()


def test_sequence_reward_and_ppo_reference_stay_frozen():
    policy=mod.MiniGPT()
    reference=mod.MiniGPT()
    reference.load_state_dict(policy.state_dict())
    reward_model=mod.SequenceReward()
    for p in reference.parameters(): p.requires_grad_(False)
    for p in reward_model.parameters(): p.requires_grad_(False)
    pair=mod.synthetic_preference_pairs(batch=6)
    assert pair[0].shape==(6,5) and pair[1].shape==(6,2)
    bt=mod.bt_pairwise_loss(reward_model,pair)
    assert torch.isfinite(bt)
    frozen=[x.detach().clone() for x in reference.parameters()]
    optimizer=torch.optim.Adam(policy.parameters(),lr=1e-3)
    loss,score,acc,ratio=mod.token_ppo_step(
        policy,reference,reward_model,optimizer,batch=6,epochs=1)
    assert all(torch.isfinite(torch.tensor(x)) for x in (loss,score,acc,ratio))
    assert all(torch.equal(before,after) for before,after in zip(frozen,reference.parameters()))
