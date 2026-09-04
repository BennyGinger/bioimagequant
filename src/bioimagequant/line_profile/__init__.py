"""Future line-profile analysis.

A line profile will measure intensity along a user-defined centreline,
optionally using a wider sampling strip to reduce pixel noise. Positions can
be represented as pixel or physical distances, or normalized from -1 to 1 to
compare samples of different sizes.

For time series, an initial line may be copied to every frame and adjusted
manually. A future implementation may use tracked segmentation masks to
translate, rotate, and resize it as a selected cell changes over time.
"""
