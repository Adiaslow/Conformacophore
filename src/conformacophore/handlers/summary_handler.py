import os
import pandas as pd

class SummaryHandler:
    def __init__(self):
        self.summaries = []

    def create_compound_summary(self, compound_id, metrics, suggestions, clusters, rmsd_matrix, cluster_info):
        # Implementation of creating the summary dictionary goes here
        pass

    def append_summary(self, summary):
        self.summaries.append(summary)

    def write_analysis_summary(self, output_dir: str):
        summary_path = os.path.join(output_dir, "analysis_summary.csv")
        df = pd.DataFrame(self.summaries)
        df = df.sort_values('compound_id')
        df.to_csv(summary_path, index=False)
        print(f"\nWrote complete analysis summary to {summary_path}")
        return summary_path
