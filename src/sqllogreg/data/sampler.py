from imblearn.over_sampling import RandomOverSampler


class Sampler:
    def resample(self, X, y):
        oversample = RandomOverSampler(sampling_strategy="minority")
        return oversample.fit_resample(X, y)
