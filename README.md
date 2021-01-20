# GMT M1 Thermal Linear Model

A linear thermal model of the GMT primary mirror based on a set of temperature Gaussian influence functions (with sigma=5cm) centered on each core of each M1 glass segment.

## Installation

First, install [Rust](https://www.rust-lang.org/tools/install), then at a terminal install the model with

`cargo install --git https://github.com/rconan/gmt_m1_thermal_influence-functions.git --branch main` 

and finally download the [data](https://s3-us-west-2.amazonaws.com/gmto.modeling/gmt_m1_thermal_influence-functions.tgz).

## Usage 

At a terminal enter: 

`tar -xzvf gmt_m1_thermal_influence-functions.tgz && cd gmt_m1_thermal_influence-functions`

then `gmt_m1_thermal_influence-functions --help` to get a description of the inputs to the model.

For example, to get 5 PSSn sample for cores temperature uniformly distributed in the range [-30,+30]mK, use: 

`gmt_m1_thermal_influence-functions -m 5 --temp-dist uniform -a 30 0`

A example of the wavefront error map is written in `wavefront.png`.
