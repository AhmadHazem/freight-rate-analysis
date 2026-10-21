import matplotlib.pyplot as plt
import pandas as pd

from .base import Analysis, AnalysisResult


def routes(df):
    return set(df['pickup'].astype(str) + '-' + df['delivery'].astype(str))


def city_coordinates(df):
    """One row per (city, lat, lon) seen as pickup or delivery."""
    pickup = df[['pickup', 'pickup_lat', 'pickup_lon']].set_axis(['city', 'lat', 'lon'], axis=1)
    delivery = df[['delivery', 'delivery_lat', 'delivery_lon']].set_axis(['city', 'lat', 'lon'], axis=1)
    return pd.concat([pickup, delivery]).drop_duplicates()


class CityRouteAnalysis(Analysis):
    """Pickup / delivery cities, routes and coordinates [INS1-INS6]."""
    title = '1.1.1 Pickup-delivery, routes and coordinates'

    def city_counts(self):
        rows = {}
        for name, df in [('train-test', self.train_test_df), ('validation', self.validation_df)]:
            pickups, deliveries = set(df['pickup']), set(df['delivery'])
            rows[name] = {
                'pickup cities': len(pickups),
                'delivery cities': len(deliveries),
                'all unique cities': len(pickups | deliveries),
                'possible routes': len(pickups) * len(deliveries),
                'rows': len(df),
            }
        return pd.DataFrame(rows)

    def unseen_cities(self):
        cities = lambda df: set(df['pickup']) | set(df['delivery'])
        return sorted(cities(self.validation_df) - cities(self.train_test_df))

    def route_comparison(self):
        tt, val = routes(self.train_test_df), routes(self.validation_df)
        return pd.Series({
            'common routes': len(tt & val),
            'only in train-test': len(tt - val),
            'only in validation': len(val - tt),
            'union of routes': len(tt | val),
        })

    def coordinates_consistent(self):
        """True when every city has exactly one (lat, lon) across both datasets."""
        coords = pd.concat([city_coordinates(self.train_test_df),
                            city_coordinates(self.validation_df)]).drop_duplicates()
        return bool((coords.groupby('city').size() == 1).all())

    def random_route(self):
        """All rows of one random route that appears more than once."""
        d = self.train_test_df.copy()
        d['route'] = d['pickup'].astype(str) + '-' + d['delivery'].astype(str)
        counts = d['route'].value_counts()
        name = pd.Series(counts[counts > 1].index).sample(n=1, random_state=self.seed).iloc[0]
        return name, d.loc[d['route'] == name, ['route', 'distance', 'market_index', 'weight', 'posted_rate']]

    def plot_route(self, name, route_df):
        plot_df = route_df.dropna(subset=['distance', 'market_index', 'posted_rate', 'weight'])

        # Dot size = weight, scaled to 20-300 so small and large weights are both visible
        w = plot_df['weight'].astype(float)
        w_range = w.max() - w.min()
        sizes = 20 + 280 * (w - w.min()) / w_range if w_range > 0 else 100

        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(projection='3d')
        ax.scatter(plot_df['distance'], plot_df['market_index'], plot_df['posted_rate'],
                   s=sizes, alpha=0.6, edgecolor='k')
        ax.set_xlabel('Distance')
        ax.set_ylabel('Market Index')
        ax.set_zlabel('Posted Rate')
        ax.set_title(f'{name}  (dot size = weight)')
        return fig

    def run(self):
        r = AnalysisResult(self.title)
        counts, route_cmp, unseen = self.city_counts(), self.route_comparison(), self.unseen_cities()
        name, route_df = self.random_route()

        r.notes += [
            f"[INS1] {counts.loc['all unique cities', 'train-test']} unique cities in train-test, "
            f"each one used as both pickup and delivery",
            f"[INS2] {len(unseen)} cities only in validation: {', '.join(unseen)}",
            f"[INS3] {route_cmp['only in train-test']} routes only in train-test, "
            f"{route_cmp['only in validation']} routes only in validation",
            f"[INS4] Coordinates consistent across both datasets: {self.coordinates_consistent()}",
            f"[INS6] Random route {name}: distance ranges {route_df['distance'].min():.1f}-"
            f"{route_df['distance'].max():.1f} mi on the same route -> city names add little",
        ]
        r.tables['city counts'] = counts
        r.tables['route comparison'] = route_cmp
        r.tables[f'random route ({name})'] = route_df
        r.figures['random_route_3d'] = self.plot_route(name, route_df)
        return r
