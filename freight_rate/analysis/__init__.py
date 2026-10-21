from .base import Analysis, AnalysisResult
from .cities import CityRouteAnalysis
from .dates import DateAnalysis
from .distributions import DistributionAnalysis
from .equipment import EquipmentAnalysis
from .features import FeatureOverview
from .outliers import OutlierAnalysis

# Notebook order
ALL_ANALYSES = [FeatureOverview, CityRouteAnalysis, DateAnalysis, EquipmentAnalysis,
                DistributionAnalysis, OutlierAnalysis]

__all__ = ['Analysis', 'AnalysisResult', 'ALL_ANALYSES', 'FeatureOverview', 'CityRouteAnalysis',
           'DateAnalysis', 'EquipmentAnalysis', 'DistributionAnalysis', 'OutlierAnalysis']
