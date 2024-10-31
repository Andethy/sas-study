import pandas as pd
from sklearn.preprocessing import MinMaxScaler
# import ace_tools as tools

def stretch_values(series, factor=1.5):
    centered = series - series.mean()
    stretched = centered * factor
    stretched_clipped = stretched.clip(-1, 1)
    return stretched_clipped + series.mean()

if __name__ == '__main__':
    file_path = '../../resources/timbre/results/average_ratings.csv'
    df = pd.read_csv(file_path)

    columns_to_normalize = ['average safety', 'average urgency']
    data_to_normalize = df[columns_to_normalize]

    scaler = MinMaxScaler(feature_range=(-1, 1))
    normalized_data = scaler.fit_transform(data_to_normalize)
    df[columns_to_normalize] = normalized_data

    df['stretched_safety'] = stretch_values(df['average safety'])
    df['stretched_urgency'] = stretch_values(df['average urgency'])

    normalized_file_path = '../../resources/timbre/results/average_ratings_normalized.csv'
    df.to_csv(normalized_file_path, index=False)

    print(f'Normalized and stretched data saved to: {normalized_file_path}')