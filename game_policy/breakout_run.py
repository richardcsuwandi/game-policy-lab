"""Paired offline gate, followed by at most two gated live edits."""
import argparse
import hashlib
import json
from pathlib import Path
import time

from game_policy.breakout import FINAL, FOLLOW, PREDICT, SELECTION
from game_policy.executor import evaluate_code
from game_policy.messages import MessageError
from game_policy.providers import live_provider

SYSTEM = """Edit a small Python paddle controller for a deterministic Breakout game.
Return only a JSON object with summary and code strings. Define exactly
choose_action(state). Return integer -1, 0, or 1 (left, stay, right).
The paddle center moves two cells per tick, then the ball moves horizontally
by vx and vertically by vy (one row). Side walls reflect at x=0 and width-1.
When descending, intercept at paddle_y. A hit within half of paddle center
returns the ball. An off-center hit sends it horizontally toward the contact
side, keeping the horizontal speed. Bricks can reflect the ball too.
State has x,y,vx,vy,paddle,width,paddle_y,half,bricks (24-bit remaining brick mask,
8 columns of 6 cells across, rows y=2..4, low bit at top left).
One life, 600 ticks. Maximize bricks cleared per serve. Use only simple Python,
no imports, helpers or private names. Keep the edit compact.
"""


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def read(path):
    return json.loads(path.read_text())


def hashes():
    package = Path(__file__).parent
    return {n: hashlib.sha256((package/n).read_bytes()).hexdigest()
            for n in ('breakout.py', 'breakout_run.py', 'executor.py', 'worker.py', 'providers.py',
                      'messages.py', 'policy_contract.py')}


def score(code, seeds, trusted=False):
    return evaluate_code(code, seeds, False, trusted, replay=True, game='breakout')


def offline(out):
    out.mkdir(parents=True, exist_ok=False)
    save(out/'config.json', dict(selection=SELECTION, final=FINAL, hashes=hashes(),
         selection_metric='mean bricks cleared in one life', step_limit=600,
         gate='positive mean paired gain and more paired wins than losses',
         max_requests=2, max_tokens=4096))
    a, b = [score(code, SELECTION, trusted=True) for code in (FOLLOW, PREDICT)]
    save(out/'follow.json', a)
    save(out/'predict.json', b)
    diff = [y['bricks']-x['bricks'] for x,y in zip(a['episodes'], b['episodes'])]
    wins, losses = sum(v>0 for v in diff), sum(v<0 for v in diff)
    gate = dict(follow=a['mean_bricks'], predict=b['mean_bricks'], wins=wins,
                ties=diff.count(0), losses=losses, proceed=sum(diff)>0 and wins>losses)
    save(out/'gate.json', gate)
    print(gate, flush=True)


def feedback(result):
    return dict(mean_bricks=result['mean_bricks'], episodes=result['episodes'])


def live(out, provider_name):
    if read(out/'config.json')['hashes'] != hashes() or not read(out/'gate.json')['proceed']:
        raise ValueError('Offline gate or source verification failed')
    provider = live_provider(provider_name, max_tokens=4096)
    dest = out/'live'
    dest.mkdir(exist_ok=False)
    save(dest/'config.json', dict(provider=provider.name, model=getattr(provider, 'model', None),
                                 requested_tier=getattr(provider, 'tier', None), max_tokens=provider.max_tokens))
    current, best = FOLLOW, score(FOLLOW, SELECTION)
    if best != read(out/'follow.json'):
        raise RuntimeError('Docker preflight disagrees with the offline evaluator')
    previous = feedback(best)
    evaluated = {'follow':FOLLOW, 'predict':PREDICT}
    stopped = None
    for edit in (1, 2):
        hint = 'First improve interception of descending balls.' if edit == 1 else 'Use the preceding trial feedback to make one compact revision.'
        system = SYSTEM + hint
        save(dest/f'prompt-{edit}.json', dict(system=system, current_policy=current, feedback=previous, generation=edit))
        event = dict(edit=edit, accepted=False)
        try:
            proposal = provider.propose(current, previous, edit, system=system)
            event.update(proposal.metrics, summary=proposal.summary,
                         request_seconds=proposal.request_seconds, usage=proposal.usage)
            (dest/f'edit-{edit}.py').write_text(proposal.code)
            start=time.monotonic()
            try:
                result=score(proposal.code, SELECTION)
            finally:
                event['evaluation_seconds']=time.monotonic()-start
            evaluated[f'edit-{edit}']=proposal.code
            save(dest/f'eval-{edit}.json', result)
            if result['mean_bricks'] > best['mean_bricks']:
                current, best = proposal.code, result
                event['accepted'] = True
            previous=dict(incumbent_mean=best['mean_bricks'], last_code=proposal.code,
                          accepted=event['accepted'], last_evaluation=feedback(result))
            event['mean_bricks']=result['mean_bricks']
        except (MessageError, SyntaxError, ValueError) as error:
            if isinstance(error, MessageError):
                event.update(error.metrics)
                if error.fatal:
                    stopped=error
            event['error']=str(error) if isinstance(error, MessageError) else type(error).__name__
            previous=dict(incumbent=feedback(best), last_failure=event['error'])
        save(dest/f'event-{edit}.json', event)
        print(edit, {k:event.get(k) for k in ('accepted','error','mean_bricks','request_seconds')}, flush=True)
        if stopped:
            break
    (dest/'champion.py').write_text(current)
    for name, code in evaluated.items():
        save(dest/f'final-{name}.json', score(code, FINAL))
    if stopped:
        raise stopped


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase',choices=('offline','live'))
    parser.add_argument('directory',type=Path)
    parser.add_argument('--provider', help='naive or a trusted module:function for the live phase')
    args = parser.parse_args()
    if args.phase == 'offline':
        offline(args.directory)
    elif args.provider:
        live(args.directory, args.provider)
    else:
        parser.error('The live phase requires --provider.')
