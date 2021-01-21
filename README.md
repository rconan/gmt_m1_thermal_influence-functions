# GMT M1 Thermal Linear Model

A linear thermal model of the GMT primary mirror based on a set of temperature influence function.
An influence function is a segment surface deformation in meters corresponding to a Gaussian temperature distribution (peak=1K, sigma=5cm) centered on one core of the segment. 
In a segment, there are as many influence functions as the number of cores.

The model compute the surface of each segment as the weighted sum of the influence function, the weights being the peak temperature of each core.
The segment surfaces (multiply by a factor 2) are assembled to form the mirror wavefront from which the PSSn is derived. 

## Installation

First, install [Rust](https://www.rust-lang.org/tools/install), then at a terminal install the model with

`cargo install --git https://github.com/rconan/gmt_m1_thermal_influence-functions.git --branch main` 

and finally download the [data](https://s3-us-west-2.amazonaws.com/gmto.modeling/gmt_m1_thermal_influence-functions.tgz).

## Usage 

At a terminal enter: 

`tar -xzvf gmt_m1_thermal_influence-functions.tgz && cd gmt_m1_thermal_influence-functions`

then, to get a description of the inputs to the model: `gmt_m1_thermal_influence-functions --help` 

```
gmt_m1_thermal_influence-functions 0.1.0
Rod Conan <rconan@gmto.org>
GMT M1 Linear Thermal Model

USAGE:
    gmt_m1_thermal_influence-functions [OPTIONS] --temp-dist <temp-dist>

FLAGS:
    -h, --help       
            Prints help information

    -V, --version    
            Prints version information


OPTIONS:
    -m, --monte-carlo <monte-carlo>             
            Number of Monte-Carlo sample [default: 1]

    -t, --temp-dist <temp-dist>                 
            Temperature distribution: constant, uniform, fan-uniform, actuator-uniform

    -a, --temp-dist-args <temp-dist-args>...    
            Temperature distribution parameters:
                 - constant        : peak[mK],
                 - uniform         : range[mK] offset[mK],
                 - fan-uniform     : sigma[m] peak[mK] range[mK] offset[mK],
                 - actuator-uniform: sigma[m] peak[mK] range[mK] offset[mK]

```

For example, to get 5 PSSn sample for cores temperature uniformly distributed in the range [-30,+30]mK, use: 

`gmt_m1_thermal_influence-functions -m 5 --temp-dist uniform -a 30 0`

A example of the wavefront error map is written in `wavefront.png`.

## Core temperature distribution
The model has four temperature distributions for the peak temperature of the segment cores:
 - constant
 - uniform
 - fan-uniform
 - actuator-uniform
