from dataclasses import dataclass, field


@dataclass
class AnalysisResult:
    """What every analysis returns, so main.py can print and save them all the same way."""
    title: str
    notes: list = field(default_factory=list)      # one-line findings
    tables: dict = field(default_factory=dict)     # name -> DataFrame / Series
    figures: dict = field(default_factory=dict)    # name -> matplotlib Figure


class Analysis:
    """Base class. Subclasses set `title` and implement run() -> AnalysisResult."""
    title = ''

    def __init__(self, train_test_df, validation_df=None, seed=None):
        self.train_test_df = train_test_df
        self.validation_df = validation_df
        self.seed = seed      # random_state for sampling; None = different each run

    def run(self):
        raise NotImplementedError
