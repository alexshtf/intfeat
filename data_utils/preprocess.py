import polars as pl


class LogSquaredBinner:
    def fit(self, df: pl.DataFrame) -> "LogSquaredBinner":
        binned = self._bin_df(df)
        self.maxs_ = binned.max()
        return self

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        return self._bin_df(df).select(
            pl.col(col).clip(0, self.maxs_[col]) for col in df.columns
        )

    def _bin_df(self, df: pl.DataFrame):
        return df.select(
            [
                self._transform_col(pl.col(col), dtype).alias(col)
                for col, dtype in zip(df.columns, df.dtypes)
            ]
        )

    def _transform_col(self, col: pl.Expr, type: pl.DataType):
        return (
            pl.when(col > 1).then(2 + col.log().pow(2).floor()).otherwise(col)
        ).cast(type)


class ClippingBinner:
    def __init__(
        self,
        max_val: int = 65536,
        quantile: float = 0.99,
        quantile_margin: float = 1.1,
    ):
        self.max_val = max_val
        self.quantile = quantile
        self.quantile_margin = quantile_margin

    def fit(self, df: pl.DataFrame) -> "ClippingBinner":
        maxs_expr = [
            pl.min_horizontal(
                (pl.col(col).quantile(self.quantile) * self.quantile_margin),
                pl.col(col).max(),
                pl.lit(self.max_val),
            )
            .round()
            .cast(dtype)
            for col, dtype in zip(df.columns, df.dtypes)
        ]
        self.maxs_ = df.select(maxs_expr)

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.select(
            pl.col(col).clip(0, pl.lit(self.maxs_[col])) for col in df.columns
        )
