single snake
6x6
unicolor
1x 20 million steps
default hyperparam
default reward function
ppo
deterministic play
-> best runs about 22-length

single snake
6x6
multicolor
1x 20 million steps
default hyperparam
default reward function
ppo
non-deterministic play
-> best runs about 19-length

multi snake
6x6
unicolor
use_bots=True
10x 2 million steps
default hyperparam
head representations 3,4 instead of 110,111
default reward function
ppo
non-deterministic play
results:
-> best runs about 29-length
-> behavior: the 2nd snake (can be the ai-snake or the bot-snake) is killed or kills itself after 2-3 steps, and then it actually becomes single snake; so 99% of training time is spent as a single snake
-> it is almost as if this 'single snake' performs better than the real single snake
-> the bot-snake seems to perform the same moves as the ai-snake, although its position is different, and therefore this seems to be a coding error; however close inspection shows that the bot-snake *does* perform differently, even if both the ai-snake and bot-snake are played in deterministic mode
-> various investigations to see if there's a coding error (e.g. swapping head representations also change the original observation due to reference semantics) did not show one

experiment 1
multi snake
6x6
unicolor
use_bots=True
1000x 20000 steps (bots get better model faster + in contrast to exploration rate in dqn, in ppo there don't seem to be hyperparameters that change throughout a learning_run)
default hyperparam
head representations 3,4 instead of 110,111
reward function:
- done=True if one snake dies; can be ai-snake or bot-snake (to spend all training time as multi player)
- if bot-snake gets a reward, it is added to reward of ai-snake (to stimulate cooperation)
ppo
deterministic play
results: 
-> snake wordt 3-7 lang; beste run was 7 lang; zie ook tensorboard plaatje en filmpje

experiment 2
multi snake
6x6
unicolor
use_bots=True
1000x 20000 steps (bots get better model faster + in contrast to exploration rate in dqn, in ppo there don't seem to be hyperparameters that change throughout a learning_run)
default hyperparam
head representations 3,4 instead of 110,111
reward function:
- done=True if one snake dies; can be ai-snake or bot-snake (to spend all training time as multi player)
- if bot-snake gets a reward, it is added to reward of ai-snake (to stimulate cooperation)
further code changes to remove random aspects (only random aspect that remains is food placement):
- no random start order; ai-snake is always started at the left side of the grid; so has become deterministic 
- no random order in which snake moves are executed; ai-snake is always the first to perform its move; so has become deterministic
- so only remaining non-deterministic aspect is where a food appears
ppo
deterministic play
results: 
-> learning stops at some point and reward goes back to 0 (see tensorboard image); daarom geen learned_model opgeslagen en geen filmpje


in multi player snake, the bot-snake seems to do the same moves as the the ai-snake. However, tested it carefylly, with both the ai-snake and the bot-snake in deterministic mode and very often the bot-snake does the same move as the ai-snake, but certainly not always

ppo: there seem to be no problem to split the learning in many small learning runs; there are no hyperparameters that change during a learning run
dqn: exploration rate changes during a learning_run, so for dqn there is a difference if you split a learning_run in many small learning_runs

