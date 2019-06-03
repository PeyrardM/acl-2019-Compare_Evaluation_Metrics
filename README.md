# acl-2019-Compare_Evaluation_Metrics

This code provides analysis and comparison of evaluation metrics for high-scoring summaries.
In particular, we find that existing evaluation metrics disagree wildly for high-scoring summaries even though they have strong agreement on standard benchmark datasets which only cover the average scoring range.

If you reuse this software, please use the following citation:

```
@inproceedings{acl-2019-compare-metrics,
    title = {{Studying Summarization Evaluation Metrics in the Appropriate Scoring Range}},
    author = {Peyrard, Maxime},
    publisher = {Association for Computational Linguistics},
    volume = {Volume 2: Short Papers},
    booktitle = {{Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (ACL 2017)}},
    pages = {(to appear)},
    month = aug,
    year = {2019},
    location = {Florence, Italy},
}
```

> **Abstract:** In summarization, automatic evaluation metrics are usually compared based on their ability to correlate with human judgments. Unfortunately, the few existing human judgment datasets have been created as by-products of the manual evaluations performed during the DUC/TAC shared tasks.
However, modern systems are typically better than the best systems submitted at the time of these shared tasks.
We show that, surprisingly, evaluation metrics which behave similarly on these datasets (average-scoring range) strongly disagree in the higher-scoring range in which current systems now operate.
It is problematic because metrics disagree yet we can't decide which one to trust.
This is a call for collecting human judgments for high-scoring summaries as this would resolve the debate over which metrics to trust. This would also be greatly beneficial to further improve summarization systems and metrics alike.


Contact person: Maxime Peyrard, maxime.peyrard@epfl.ch

