# Learning notes: shot quality, in plain English

Julian, this is the version you study so you can explain the project without reading the code out loud. The README has the results. This file has the ideas, in the order you would defend them.

Every number below is from the pipeline that wrote `data/processed/metrics.json`. If you rerun and the fourth decimal moves, talk about the rounded figures in the README.

## 1. The question

A shot has two parts.

1. **What was the attempt worth before it went up?** That is shot quality, or shot selection when you average it for a player.
2. **Did it go in more often than attempts like that go in?** That is shot making.

FG% mixes those two. A center who only dunks can shoot 70% and be an ordinary finisher of dunks. A forward who lives on long twos can shoot 45% and be an excellent shooter of hard shots. Expected points separates the mix from the making.

The output of the model, for one shot, is a probability. Call it xFG%. Multiply by 2 or 3 and you have expected points, xPTS.

- Made three, model said 40%: actual points = 3, xPTS = 1.20, points over expected = **+1.80**.
- Missed three, model said 40%: actual points = 0, xPTS = 1.20, points over expected = **−1.20**.

Add those up for a player. Divide by attempts. Multiply by 100 so you can say “plus 16 points per 100 shots” instead of “plus 0.16.”

## 2. Where the rows come from

Each row is one field-goal attempt in the regular season, 2023-24 through 2025-26. 1,230 games a season, which is a full 30-team, 82-game schedule (30 × 82 / 2 = 1,230). About 219,000 shots a season. League FG% sits between 46.7% and 47.4%.

The live NBA stats API (`ShotChartDetail` through `nba_api`) timed out from the machine that built this. The files in `data/raw/` are the SportsDataverse mirror of that same feed. `scripts/pull_shots.py` still knows how to call the API, cache each team, and wait between calls. On a laptop where stats.nba.com answers, `--source api` uses that path. The columns are renamed onto one layout so the rest of the code does not care which source you used.

A few fields look tempting and are traps.

- `action_type` in the mirror means “Made Shot” or “Missed Shot.” That is the answer key. The shot description is `sub_type` (“Cutting Dunk Shot”, “Step Back Jump shot”).
- The description text contains “MISS” on misses. Do not bag-of-words it.
- The score columns are filled in only when the shot is made, and the gap between two makes includes free throws. That is not the score before the shot. It was left out. Say that out loud if someone asks “why didn’t you use game context?” The honest answer is “the field I had was not the pre-shot score, and the post-shot score would have leaked the make.”

Heaves (40 feet and farther, 1,187 shots) are flagged and kept in the database so you can count them, then dropped from the model. A 70-foot shot at the buzzer is not shot selection.

Full names come from the player list that ships inside `nba_api`. That file is local. Every id in these three seasons matched, which is worth knowing because the mirror itself only stores a last name. “Williams” is not an analysis key. The id is.

## 3. Features, and the leakage rule

Leakage means the model sees something that would not have been known when the shot went up, or something that is just a copy of the thing you are predicting.

Allowed, because a coach could see them as the ball left the hand:

- where (x, y, distance, zone, 2 vs 3)
- what kind of shot, in a coarse family
- how much time was left in the period, and which period

Not allowed:

- made or missed, points scored, the word MISS
- player id and team id

The second one surprises people. It is not classic label leakage. It is still the wrong model. If the model knows it is Durant shooting, part of “expected” FG% is Durant’s own history. Points over expected then shrinks toward zero and you have built a player model while calling it a shot model. The code keeps the allowed columns in `FEATURE_COLUMNS` and the banned ones in `FORBIDDEN_COLUMNS`, and a test fails if they overlap.

Zones are rebuilt from the coordinates because this mirror does not include the NBA’s zone names. The rules are in `src/shotquality/features/build.py` and they are worth being able to draw on a whiteboard:

- Restricted area: within 4 feet of the rim.
- Paint, non-restricted: inside the 16-foot lane, outside that 4-foot arc, no farther than the free-throw line.
- Corner three: a 3, with the shooter outside the 22-foot corner line and below the spot where that line meets the arc (about 8.95 feet up from the rim).
- Above the break: every other 3.
- Midrange: the 2s that are none of the above.
- 40+ feet: backcourt / heave, then dropped.

On 2023-24, that three-point geometry matches the official 2/3 flag on 99.99% of shots. Restricted-area FG% over three seasons is 66.6%. If someone asks “how do you know your zones are real?”, those two checks are the answer.

Action families are dunk, hook, tip, floater, layup, jumper. Order matters: “Tip Dunk Shot” is a dunk. “Driving Floating Jump Shot” is a floater, not a jumper. Step-backs stay in jumper. Give every move its own rate and you will call a specialist’s move “shot quality.”

## 4. Why the split is by season, not by row

If you randomly hold out 20% of rows, two shots from the same game land on both sides. The model has seen the pace, the opponent, the whistle, and the rest of that night. The test score is then partly an in-sample score.

This project trains on 2023-24 and 2024-25 and tests on every 2025-26 regular-season shot (219,100 after dropping heaves). Those games had not been played when the training shots happened. No game id is in both sets. There is a test for that.

Inside the training seasons, the last 20% of 2024-25 game ids (246 games, 43,691 shots) are a validation slice. They are used for two decisions only:

1. When the boosted model should stop adding trees (it stopped at 107).
2. Which model the leaderboard uses (lowest validation log loss).

Then every model is refit on all of the training seasons, with that tree count locked, and scored once on 2025-26. You do not pick the winner by looking at the test set and then quoting the test set. In this run the validation winner and the test winner were the same model. If they had disagreed, the leaderboard would still follow validation, and you would say so.

Game ids follow the schedule as it was written. A postponed game keeps its id, so the validation cut is not a pure date cut. Say that before someone else does. It does not affect the test season.

## 5. The baseline you have to beat

The zone model says: a restricted-area shot goes in at the training-set restricted-area rate, a midrange shot at the midrange rate, and so on. Unknown zones would get the overall training rate. There is no player in it.

On the holdout its log loss is **0.6569** and its Brier score is **0.2321**. AUC is 0.637. It is well calibrated (calibration error 0.005) because a zone rate estimated on two seasons is almost the same rate the next season produces.

If logistic regression or boosting cannot beat 0.6569, they are not earning their complexity. Both do.

## 6. Logistic regression

You model the log-odds of a make:

\[
\log\frac{p}{1-p} = \beta_0 + \beta_1 x_1 + \cdots
\]

`p` is the probability the shot goes in. The log-odds can be any real number, so a straight line is a legal model. The probability you get back is squeezed between 0 and 1 by the logistic function. That is the whole trick. You are not fitting a probability with ordinary least squares and hoping it stays inside 0 and 1.

L2 regularization (`C = 1`) shrinks coefficients toward zero so a rare action dummy cannot take a wild value. Numeric features are standardized so distance in feet and seconds on the clock live on a similar scale. Categories are one-hot encoded. A category the training set never saw is ignored rather than crashing.

Holdout log loss **0.6437**, Brier **0.2269**, AUC 0.654.

Do not interpret the distance coefficient as “the effect of taking one step back.” Distance, x, y, and zone describe the same spot three ways. The predictions can still be good. The individual coefficients are not a clean story. The zone table and the distance chart are the story.

## 7. Log loss and Brier score

Both score a probability. Accuracy does not, and it will embarrass you if you lead with it.

**Log loss** for one shot is \(-\log(p)\) if it went in and \(-\log(1-p)\) if it missed. Average that over shots. A confident wrong answer (p = 0.99 on a miss) is punished hard, because log(0.01) is a large negative number. A 50/50 guess on every shot scores \(-\log(0.5) \approx 0.693\). Our zone model at 0.6569 is clearly better than a coin flip. The boosted model at 0.6387 is better than the zone model by 0.018. Small, and it shows up again in Brier score, and it showed up first on the validation season.

**Brier score** is the average of `(probability − outcome)²`. Outcome is 0 or 1. A perfect model scores 0. A coin flip scores 0.25. Ours: zone 0.2321, logistic 0.2269, boosting 0.2247.

Why both? They usually agree, and when they do you can stop worrying that you picked a weird metric. Log loss cares more about being wrongly confident. Brier is easier to say out loud (“average squared error of the probability”).

AUC (0.637 zone, 0.654 logistic, 0.660 boosting) asks a different question: if you pick a made shot and a missed shot at random, how often does the model give the make the higher probability? Useful, and not the objective. A model can rank shots well and still be miscalibrated, which would ruin expected points. We require both.

## 8. Calibration

Calibration means: among shots where the model says 40%, about 40% go in.

The chart plots predicted probability on x and the actual make rate on y. A perfect model sits on the diagonal. The holdout boosted model does, across the range that matters:

- says 37%, shots go in 36.1% (93,670 shots)
- says 44%, shots go in 44.8% (64,728)
- says 75%, shots go in 75.1% (13,696)
- says 93%, shots go in 94.2% (6,690)

The single summary is expected calibration error: the average gap between predicted and actual, weighted by how many shots are in the bin. Zone 0.0050, boosting 0.0067, logistic 0.0086. The zone model wins this particular trophy because it only says a few numbers and those numbers are the historical rates. The boosted model is a close second and assigns different probabilities to shots inside the same zone, which is why its log loss is better.

Mean predicted FG% on the holdout is 47.2%. Actual FG% is 47.1%. The model is not drifting high or low overall. It is a little high on above-the-break threes (1.074 expected points vs 1.051 actual). Say that. It is the honest blemish.

## 9. Gradient boosting, without the mythology

Start with a simple model. Look at which shots it gets wrong. Fit a small tree to those errors. Add the tree, gently (learning rate 0.05, so each tree only moves the prediction a little). Repeat. Stop when a set of games the model is not training on stops improving. That stopping point was 107 trees, chosen on the 2024-25 validation games, then the model was refit on all training shots with 107 trees.

Each tree is shallow in the sense that matters here: at most 31 leaves, and a leaf must contain at least 200 shots. The model cannot memorize a single weird make.

“Boosting” is the adding-up. “Gradient” means each new tree is aimed at the remaining error of the loss you care about (here, log loss). You do not need to derive the gradient in an interview. You need to say what is added, what stops it, and which data the stopping decision saw.

Holdout log loss **0.6387**, Brier **0.2247**, AUC 0.660. It won validation, so it is the model behind the leaderboard and the app’s default.

Feature gain, which is “how much did splits on this column reduce the loss,” is 60.5% distance, 25.8% action family, 5.7% zone. Gain is not a causal effect. Distance wins partly because zone and the 2/3 flag repeat it.

## 10. Shrinkage, with the number from this run

A player who is +25 points per 100 shots on 40 attempts has mostly told you about variance. The residual on one shot has a standard deviation of **1.17 points**. The standard error of a mean over n shots is \(1.17 / \sqrt{n}\). Per 100 shots, multiply by 100.

| Attempts | Standard error, points per 100 |
| ---: | ---: |
| 100 | 11.7 |
| 300 | 6.8 |
| 1,000 | 3.7 |

So the minimum of 300 attempts still leaves a fat error bar. The bar chart shows it.

Shrinkage goes further. Assume each player has a true rate, those true rates cluster around the league, and what we observe is the true rate plus noise. The formula that falls out is a weighted average:

\[
\text{shrunk} = \frac{n}{n+k}\times\text{observed} + \frac{k}{n+k}\times\text{league}
\]

`k` is in units of shots. We estimate it from the data rather than picking a round number and hoping:

\[
k = \frac{\sigma^2}{\tau^2}
\]

`σ²` is the variance of a single shot’s residual (within player). `τ²` is how spread out the true player means are. Method of moments: the spread you see in player averages equals the true spread plus `σ² / n`. Subtract that noise term. What is left is `τ²`.

For shot making, that estimate is **k = 176**. A player with 176 shots is pulled halfway to the league. Luke Kennard’s raw +27.1 per 100 on 443 shots becomes **+19.2**. The weight is 443 / (443 + 176) ≈ 0.72, and 0.72 × 27 is about 19. Jokić on 1,130 shots keeps more of his +20.7 and lands at **+17.9**.

For shot selection, k is only **8.6**. Players really do take different shots, and xPTS does not bounce around the way make/miss does. Gobert’s +33 barely moves. You should be able to say why the two k’s differ. That question is coming.

Teams are not shrunk with the player k. A team residual is a different parameter, and at 7,000 shots the standard error is already about 1.4 points per 100. Denver at +7.6 is roughly five standard errors from zero. The middle 18 teams are not.

## 11. How to talk about a player

Two numbers, both per 100 shots.

- **Selection:** the player’s average xPTS, minus the league’s 1.095, times 100. Positive means a more valuable diet.
- **Making:** points over expected per 100, then shrunk.

The scatter plot is those two axes. Corners:

- Gobert, Gafford: valuable diet, ordinary or slightly negative making once dunks are priced in.
- Durant, DeRozan, Jokić: harder diet, clearly positive making.
- Tomlin: valuable diet, shots did not go in.
- A lot of role players: inside ±2 per 100 on both, which the app calls “about average” so we do not narrate noise.

Jaylen Brown is the volume example in the figure and the default in the app: 1,543 attempts, selection −9.4, shrunk making +3.9. The dots are colored by xFG%, not by whether the shot fell. If you color by make/miss you have drawn a shot chart. If you color by xFG% you have drawn the model.

## 12. The SQL, and why it is there

DuckDB holds `shots` (every cleaned attempt, heaves flagged) and `predictions` (the holdout, with probabilities).

The queries in `src/shotquality/analysis/sql_queries.py` answer:

- shots, games, FG%, three-point rate, heaves, by season
- FG% and points per shot by zone, heaves excluded
- FG% by action family
- each team’s rim rate, midrange rate, and three-point rate in a season
- per player on the holdout: attempts, FG%, xFG%, points, expected points, and the team they shot the most for

That last one uses `QUALIFY`, which filters a window function without wrapping it in another subquery. “Keep the team with the most shots for this player, break ties by name.” Counting is SQL. The prior strength `k` needs every shot’s residual, so the shrinkage step stays in Python where the formula is readable. You can say that split out loud. It is a reasonable design, not a dodge.

## 13. Questions an interviewer will ask

**Why not a random train/test split?**
Shots from the same game share a defense, a whistle, and a score. A random split puts that night on both sides. The test set here is a later season. No game is in both.

**Isn’t 0.018 of log loss a tiny improvement?**
Yes, the probabilities move a little. They move in the same direction on Brier score, they moved in the same direction on the validation season before the test season was opened, and the zone model was already a serious baseline. Expected points needs the sharper probability inside a zone (a dunk is not a hook). The leaderboard is that difference accumulated over hundreds of shots, which is why Durant can be +249 points on the season.

**Why is the zone model better calibrated and worse at log loss?**
It only predicts a few rates, and those rates are stable, so the average in each bin matches. It cannot tell two shots in the same zone apart. Log loss rewards that distinction. Calibration error does not, once the bin is as wide as a zone.

**Walk me through log loss on one shot.**
Missed shot, model said 0.90. Loss is −log(0.10) ≈ 2.3. If the model had said 0.60, the loss is −log(0.40) ≈ 0.92. Same miss, much smaller penalty, because the model was less sure. Average that over 219,100 shots.

**Why shrink, and why k = 176?**
Because the standard deviation of one shot’s point residual is 1.17, so a 300-shot mean still has a standard error near 7 points per 100. k = σ² / τ² from a method-of-moments estimate on the holdout residuals. 176 is “how many shots of noise equal the real gap between players.” I did not pick 176 because it made the leaderboard look nice. Shot selection gets a different k (8.6) because shot diets genuinely differ and are less noisy.

**What would change if you put player id in the model?**
xFG% would partly be that player’s own shooting. Points over expected would collapse toward zero for veterans and look huge for rookies the model had barely seen. You would no longer be measuring shot making.

**Durant is +16 per 100 and his shot quality is −10. Is he a bad selector?**
He takes harder shots than the league and more of them go in than those shots are worth. Whether the hard shots are a good idea depends on the alternative, which this row does not contain. The model prices the shot he took. It does not know what a drive would have produced. Say that limit before someone else does.

**Gobert leads shot quality and is ordinary at shot making. Does the model think he can’t finish?**
It thinks a dunk goes in 88.6% of the time no matter who dunks it. Relative to other dunks and layups, he is around average. His entire surplus in this framework is that he almost only takes those shots. If the question is “who converts difficult shots,” look at Durant and DeRozan. If the question is “whose shot diet is already valuable,” look at Gobert. I grouped actions so those two questions would not be the same number. I can also show the version with action type removed. I would not hide the choice.

**Why not use defender distance?**
It is not in this feed. Location is a proxy for contest and a bad one. An open corner three and a contested one share a coordinate. That is the first feature I would add with tracking data, and it would change both the probabilities and who shows up as a shot-maker.

**Your logistic model uses distance and zone together. Isn’t that collinear?**
Yes. Predictions can still be accurate under L2. I do not quote a distance coefficient. The interpretable objects are the zone table, the distance chart, and the gain ranking, with the collinearity stated next to the ranking.

**How did you pick the boosted model’s complexity?**
Validation log loss, on games inside 2024-25, with early stopping. Minimum 200 shots in a leaf so a leaf cannot be one player’s heater. Then refit on all pre-2025-26 shots with the chosen tree count. The test season was not used to pick trees or to pick the model.

**A team is +2 per 100. Do you believe it?**
The standard error is about 1.4 per 100. +2 is inside two standard errors. Denver at +7.6 is not. Twelve of 30 teams clear two standard errors. I would not rank 15th versus 16th.

**Why is 2025-26 missing most heaves?**
The mirror has 56 shots from 40 feet and beyond, against 510 and 621 in the previous two seasons. Shots inside 30 feet, and the 1,230 games, look complete. Heaves are excluded from the model in every season, so the hole does not move xFG%. I would rather say the hole is there than quietly drop the season.

**Could you build this in Excel?**
The zone baseline, yes: a pivot of FG% by zone on the training seasons, `VLOOKUP` it onto the holdout, multiply by 2 or 3. That is why it is the baseline. Logistic regression and boosting are the part Excel does not do honestly at this row count. The SQL is the part you would also do in a team database. The shrinkage formula is a column: `= n/(n+176) * raw`.

**What breaks first if a team actually used this?**
Contest. Then sample size, if someone ranked a 50-shot hot streak. Then treating one season of points over expected as a contract input. The standard errors are in the table so that conversation can happen with the number on the page.

## 14. What I would improve next, in the order I would say it

1. Defender distance and shot clock, from tracking.
2. Several seasons of out-of-fold predictions, so the leaderboard is not one year.
3. Shrink inside the model instead of after it.
4. A side-by-side with action type removed, so diet and finishing can be shown as a pair.

## 15. A path through the code, if they ask you to open the laptop

1. `src/shotquality/data/pull.py` — source, cache, the columns that are banned on the way in.
2. `src/shotquality/features/build.py` — zones, action families, `FEATURE_COLUMNS`, `split_shots`.
3. `src/shotquality/models/baseline.py` — the spreadsheet model.
4. `src/shotquality/models/train.py` — validation, refit, one test score.
5. `src/shotquality/models/evaluate.py` — log loss, Brier, xPTS.
6. `src/shotquality/models/shrinkage.py` — k, and the weighted average.
7. `src/shotquality/analysis/sql_queries.py` — the SQL you are willing to read aloud.
8. `notebooks/shot_quality_analysis.ipynb` — the same path, with the charts.
