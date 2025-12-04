# Attention Map Analysis: Understanding Patterns Across Different Time Series Distributions

## 1. HeavyTailedAR1

The attention maps for HeavyTailedAR1 show **moderate diagonal patterns with occasional concentrated spikes**. This reflects the nature of AR(1) processes with Student-t noise: while most transitions are smooth and local (creating diagonal attention patterns), the occasional massive outliers cause the model to develop specialized attention heads that "overreact" to extreme values. Some heads show strong local attention (bright diagonal bands), while others exhibit more uniform patterns, suggesting the model learns to balance between tracking the autoregressive structure and detecting anomalous spikes. The attention is more spread out than pure local attention because the heavy tails create uncertainty about whether a large value is signal or noise, requiring the model to attend to broader context to make this determination.

## 2. GARCH11

GARCH11 attention maps display **clustered, block-like patterns** that correspond to volatility regimes. The model learns to identify periods of high and low volatility, creating attention patterns where certain time periods (corresponding to volatile episodes) receive concentrated attention across multiple positions. You'll see bright horizontal or vertical bands that represent the model attending heavily to turbulent periods. The attention is less uniform than stationary processes because volatility clustering means the model needs to "remember" recent volatility states - hence attention patterns that group together time periods within the same volatility regime. Some heads specialize in detecting volatility transitions, showing sharp boundaries between high and low attention regions.

## 3. JumpDiffusion

JumpDiffusion produces attention maps with **scattered hotspots and disrupted diagonal patterns**. The model struggles to maintain consistent local attention because sudden jumps break the continuity of the time series. Attention heads develop two strategies: some maintain weak diagonal patterns for the smooth diffusion component, while others create isolated bright spots that "catch" jump events. The attention is more fragmented than other processes because jumps are unpredictable and discontinuous - the model can't rely on smooth transitions, so it learns to attend broadly to detect potential jump locations. The pattern shows the transformer's difficulty in handling discontinuous events, with attention "jumping around" rather than following smooth temporal dependencies.

## 4. RegimeSwitching

RegimeSwitching attention maps show **distinct block structures** corresponding to different regimes. The model learns to identify regime boundaries, creating attention patterns where positions within the same regime attend strongly to each other, but attention drops at regime transitions. You'll see rectangular blocks of high attention separated by darker boundaries - this is the model learning the hidden Markov structure. Some heads specialize in detecting regime changes (showing attention spikes at transition points), while others maintain consistent attention within regimes. The block structure reflects the abrupt nature of regime switches - the model needs to "reset" its attention when the underlying distribution changes, creating these clear boundaries in the attention maps.

## 5. TrendBreaks

TrendBreaks attention maps exhibit **segmented diagonal patterns with breaks**. The model maintains local attention within each trend segment (creating diagonal bands), but these bands are interrupted at trend break points. You'll see multiple diagonal regions separated by vertical or horizontal discontinuities where attention patterns shift abruptly. Some heads track the trend within segments (strong diagonal patterns), while others detect break points (showing attention concentration at structural change locations). The segmented nature reflects how trend breaks create "memory resets" - the model can't rely on long-term trends, so it learns to attend locally within segments and identify when the trend changes direction.

## 6. RandomWalk

RandomWalk attention maps show **weak, diffuse patterns** with minimal structure. Since random walks have no mean reversion and each step is independent, the model struggles to find meaningful patterns to attend to. The attention is more uniform and less concentrated than other processes because there's no predictable structure - no cycles, no trends, no regimes. Some heads may show weak diagonal patterns (attending to recent history), but overall the attention is spread out, reflecting the model's uncertainty about what information is useful. The low concentration indicates the transformer recognizes the non-stationary, memoryless nature of random walks - it can't learn much from past values because they don't predict future behavior in a structured way.

## 7. SeasonTrendOutliers

SeasonTrendOutliers attention maps display **periodic patterns overlaid on diagonal structures**. The model learns the underlying seasonal cycle, creating attention patterns that repeat at the seasonal frequency - you'll see horizontal or vertical stripes corresponding to seasonal phases. The diagonal component captures the trend, while the periodic component reflects seasonal dependencies. Some heads specialize in the seasonal pattern (showing clear periodic stripes), while others handle outliers (creating isolated bright spots). The combination of periodic and diagonal patterns shows how the model decomposes the signal into trend, seasonality, and anomalies - each requiring different attention strategies.

## 8. MultiSeasonality

MultiSeasonality attention maps show **complex overlapping periodic patterns**. The model learns multiple seasonal frequencies simultaneously, creating attention maps with intricate patterns that combine different periodicities. You'll see multiple sets of stripes or waves at different frequencies - this is the model attending to different seasonal components. Some heads specialize in shorter cycles (daily patterns), while others capture longer cycles (weekly patterns). The attention is more structured and concentrated than single-seasonality processes because the multiple predictable patterns provide more information for the model to exploit. The overlapping patterns reflect the complexity of the signal - the model needs to track multiple cycles simultaneously.

## 9. OneOverFNoise (1/f Noise)

OneOverFNoise attention maps exhibit **long-range, persistent attention patterns**. The 1/f noise has long memory (correlations decay slowly), so the model learns to attend to distant time points, not just local neighbors. You'll see attention patterns that extend far from the diagonal - bright regions that connect distant time positions. The attention is more spread out and less concentrated than processes with short memory because long-range dependencies mean every position is potentially relevant. Some heads may show weak diagonal patterns (local structure), while others show horizontal or vertical bands (attending to specific time scales). The persistent, long-range attention reflects the fractal nature of 1/f noise - correlations exist across all time scales, requiring the model to maintain attention over extended periods.

---

## Key Insights

1. **Local vs. Global Attention**: Processes with short memory (AR, Random Walk) show more diagonal/local attention, while long-memory processes (1/f noise) show extended attention patterns.

2. **Structural Breaks**: Processes with regime changes or trend breaks create attention maps with clear boundaries and segmented patterns.

3. **Periodic Patterns**: Seasonal processes produce attention maps with periodic stripes or waves corresponding to seasonal frequencies.

4. **Volatility Clustering**: GARCH processes create block-like attention patterns corresponding to volatility regimes.

5. **Discontinuities**: Jump processes and outliers create scattered, fragmented attention patterns as the model struggles with discontinuous events.

6. **Predictability**: More predictable processes (seasonality, trends) produce more structured attention maps, while unpredictable processes (random walk) show diffuse, uniform attention.

