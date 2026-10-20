# Defense Q&A Bank

Short, evidence-backed answers to the questions we expect in the presentation and consultation. Every number comes from the formal run `experiment-v2-002`, the second-period run `experiment-v3-late2017-001`, or their analysis layers (`artifacts/analysis/*/report_facts.json`). Sections refer to `docs/final-report.en.md`.

---

## Problem and data

### Q1. Olist is a well-used Kaggle dataset. What is new here?
Most public notebooks predict lateness with random splits, use delivery timestamps as inputs and report accuracy. We instead model one operating decision: an evening review queue with a fixed daily budget, using only information available at purchase. On top of that:
- We evaluate strictly in time order.
- We keep the 2,971 orders with unknown outcomes visible instead of treating them as on time.
- We report results as orders reviewed, delays caught and false alarms.
- We ask whether acting at purchase time is worth it at all, which leads to the shipping-limit checkpoint.

### Q2. Why define "late" by calendar day?
Promised dates are stored at midnight. With a clock-time comparison, 1,292 orders that arrived during the promised day would count as late, although a customer sees them as on time. The calendar-day label gives 6,534 late orders (6.77%), and the strict label gives 7,826 (8.11%). Re-running the backtest with the strict label gives 150 caught orders at 8.0% precision and 14.7% recall, so the conclusion does not depend on the choice.

### Q3. Why not drop or zero-fill orders that were never delivered?
- **Zero-filling** would count 1,107 shipped-but-never-confirmed orders, plus the canceled and unavailable ones, as successful deliveries.
- **Dropping them** would hide that they still occupy places in the review queue.

We exclude them from training and accuracy, but keep them in the operational queue. In the backtest they took 45 of 1,922 alerts, so true precision lies between 6.0% and 8.4%.

### Q4. Why is random cross-validation wrong here?
The late rate moved from 5.6% in 2017 to 19.0% in March 2018 and 1.2% in June 2018. Random splits would let a January model learn from March outcomes. Outcomes also arrive with a delay. At our selection date, 104 April orders were still undelivered, and 89 of them ended late, so April looked like 3.24% late but was really 4.50%. We only train on outcomes known at each cutoff.

---

## Method

### Q5. What exactly is the selected rule?
Score = route late rate × promise tightness.
- **Route late rate:** the smoothed share of late deliveries among earlier delivered orders on the same seller-state to customer-state route. It is shrunk toward the overall rate with a weight of 5 orders.
- **Promise tightness:** the median promised lead time in training (24 days) divided by the order's own promised lead time. The lead time is first limited to 3–90 days, and the ratio to 0.5–2.

It has two inputs and two fitted numbers. The identifier `rule_region_lead` is historical; the rule uses routes, not a region table.

### Q6. Why only logistic regression and shallow trees?
The course stresses appropriate, interpretable methods, and the question is whether any learned model beats simple rules. We fixed a small grid in advance: 8 logistic regressions and 16 trees over 74 purchase-time features, plus 7 payment features tested separately. That kept the comparison honest. With a weak signal and a shifting regime, larger models would mostly add variance.

### Q7. How did you choose the winner without looking at the test window?
The selection rule was fixed in the configuration before any run: highest mean daily top-10% recall over four monthly validation folds (January–April 2018), and a model must beat the best rule. The rule scored 25.2%, against 21.0% for the best logistic regression. The backtest (26 May to 29 August 2018) was scored once afterwards.

### Q8. Was the backtest really unseen?
Not entirely, and we say so:
- The project audit computed descriptive statistics of the window.
- Two smoke runs with a reduced grid scored the window shortly before the formal run, and pipeline code was edited in between.

On the other hand, the configuration, candidate grid and selection rule had been fixed before either smoke run, and a later smoke re-run on the final code reproduced the earlier smoke predictions exactly. We call the window a diagnostic holdout, not a blind test.

### Q9. How did you quantify uncertainty?
We used a weekly block bootstrap: whole calendar weeks are resampled, because delays cluster by week. The backtest has 15 weeks. With 2,000 draws:
- recall 12.7% to 27.0%
- precision 4.2% to 8.4%
- ROC-AUC 0.586 to 0.706

The first formal run used 400 draws because the number was hard-coded. The fixed code re-ran it with the configured 2,000 draws, and the predictions were byte-identical.

---

## Results and interpretation

### Q10. Is 17.4% recall at 6.2% precision any good?
It is better than chance: precision lift is 1.78, and the whole precision interval lies above the 3.49% base rate. Operationally it is weak, because the team reviews 16.1 orders per late order found. That is why we do not recommend a human review queue at purchase time.

### Q11. Why did machine learning not help?
Most of the usable signal at purchase time is "where is it going" and "how much slack does the promise leave", and the rule encodes both directly. The models had to learn them from noisy data in a changing regime. They were also unstable: the best logistic regression caught 41.5% of late orders in April but 2.9% in February. In the backtest, none of the 24 models beat the rule; the best reached 15.3%. Payment features made no material difference.

### Q12. The shortest-promise rule did best in the backtest. Why not use it?
- **It was the worst candidate in validation** (4.9% recall). Its backtest success comes from a pattern that appeared later: promises of 10 days or less rose from 1.2% to 15.0% of orders and were late 11.6% of the time.
- **Its edge is not statistically clear.** The paired bootstrap interval for the recall difference runs from −13.4 pp to +13.9 pp.
- **Switching after seeing the test would invalidate the test.**

The right response is to monitor and re-select on a fresh period. The pre-registered late-2017 test vindicated this: there the shortest-promise rule caught only 4.2% of late orders, worse than random.

### Q13. What actually changed between the periods?
- **Early 2018:** cross-state routes were the problem (14.2% late against 4.9% within a state).
- **The backtest:** short local promises were the problem. São Paulo to São Paulo orders promised within 10 days produced 41.4% of all late orders.

The data are consistent with a change in how promise dates were set for local orders, but they cannot tell us why it happened.

### Q14. Can the scores be read as probabilities?
No. The rule under-predicted in the validation months (5.5% average score against 10.5% observed) and over-predicted in the backtest (7.5% against 3.5%, a ratio of 2.16). They are only useful for ordering orders within a day, which is how the daily quota uses them.

### Q15. What is the shipping-limit signal, and why trust it?
The signal flags an order if the seller has not handed it to the carrier by the shipping-limit deadline. In the backtest:
- It raised 1,575 alerts (about 16 a day), with 19.4% precision and 45.4% recall.
- That is 3.1 times the precision of the purchase-time queue. The precision gain has a bootstrap interval of 4.3 to 20.1 points.
- Orders that ended late still had a median of 3.2 days before the promise when flagged.

We designed it after seeing the 2018 results, so we then tested it on the untouched late-2017 window:
- It reached 28.9% precision and 34.0% recall there.
- It beat the purchase-time queue's precision by 7.3 points (interval 1.4 to 13.6 points).

Its recall depends on where delays arise. In the January–April 2018 months it was 19.4%, because delays then came from carrier transit, which this signal cannot see. It still needs a pilot to show that acting on it prevents delays.

### Q16. Do the cost numbers mean Olist would save money?
No. They show break-even conditions under three scenarios that were fixed in the configuration before any run.
- In the low-delay 2018 window, a purchase-time queue at 6.2% precision pays only if a caught delay is worth about 16 times a contact. That holds only in the most optimistic scenario: +31.9 per 1,000 orders.
- In the high-delay late-2017 window (21.6% precision), the same queue earns +69.4 in the moderate scenario.
- The shipping-limit signal clears the moderate scenario in both periods (+49.0 and +128.1).

Whether any action prevents a delay can only be measured with a randomized pilot.

---

## Limitations

### Q17. Did you try to improve the model, and how did you avoid fooling yourselves?
Yes. We froze three new rules (60-day route history, a censoring-aware 60-day route rate and promise slack), a model feature set with the same measures, and censoring-aware labels in new configuration files. Only then did we score them on late 2017, a period no model had seen.
- **Result:** the 60-day route rule won selection and beat the original rule by 7.7 points of recall (interval 4.5 pp to 12.8 pp).
- **Models:** still no gain over the rules. The best one fell 0.4 percentage points short on validation.
- **Hindsight winners:** promise slack, which would have been best in hindsight on 2018 (27.7%), failed on late 2017 (8.3%).

We did not tune anything on the 2018 window.

### Q18. What are the biggest weaknesses?
- **One marketplace, one era.**
- **A label defined by Olist's own promise algorithm**, which appears to have changed.
- **Short test windows** (96 and 67 days) with wide intervals. The second window is unseen, but it comes earlier in time than 2018.
- **A diagnostic rather than blind holdout.**
- **Post hoc analyses** for the most actionable finding.
- **Purely associational evidence.**

We report each of these openly in Section 6 of the report.
