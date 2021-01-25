# GMT M1 Thermal Linear Model

A linear thermal model of the GMT primary mirror based on a set of temperature influence function.
An influence function is a segment surface deformation in meters corresponding to a Gaussian temperature distribution (peak=1K, sigma=5cm) centered on one core of the segment. 
In a segment, there are as many influence functions as the number of cores.

The model compute the surface of each segment as the weighted sum of the influence function, the weights being the peak temperature of each core.
The first 27 eigen modes of each segment is then removed from the thermal segment surface deformation.
The segment surfaces (multiply by a factor 2) are assembled to form the mirror wavefront from which the PSSn is derived from. 

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
    -h, --help       Prints help information
    -V, --version    Prints version information

OPTIONS:
    -b, --band <band>                           Photometric band (V: 0.5micron or H: 1.65micron) [default: V]
    -m, --monte-carlo <monte-carlo>             Number of Monte-Carlo sample [default: 1]
    -t, --temp-dist <temp-dist>                 Temperature distribution: constant, uniform, fan-uniform, actuator-
                                                uniform
    -a, --temp-dist-args <temp-dist-args>...    Temperature distribution parameters:
                                                     - constant        : peak[mK],
                                                     - uniform         : range[mK] offset[mK],
                                                     - fan-uniform     : sigma[m] peak[mK] range[mK] offset[mK],
                                                     - actuator-uniform: sigma[m] peak[mK] range[mK] offset[mK]
```

For example, to get 5 PSSn sample for cores temperature uniformly distributed in the range [-30,+30]mK, use: 

`gmt_m1_thermal_influence-functions -m 5 --temp-dist uniform -a 30 0`

The wavefront error map corresponding to the 1st sample is written in `wavefront.png`.
The temperature field for segment #1 and segment #7 for the 1st sample is written in `temperature.distribution.png` (triangles show fan locations, actuator locations are represented with black dots).

## Core temperature distribution
The model has four temperature distributions for the peak temperature of the segment cores:
 - constant
 - uniform
 - fan-uniform
 - actuator-uniform

### Constant temperature distribution
The same peak temperature is applied to all the cores.

Example:

[gmt_m1_thermal_influence-functions --temp-dist constant --temp-dist-args 30](https://github.com/rconan/gmt_m1_thermal_influence-functions/tree/main/tests/constant_30)

### Uniform
The core peak temperature is uniformly distributed in [-range,+range] + offset.

Example:

[gmt_m1_thermal_influence-functions -m 5 --temp-dist uniform --temp-dist-args 30 0](https://github.com/rconan/gmt_m1_thermal_influence-functions/tree/main/tests/uniform_30_0)

### Fan-Uniform
A Gaussian-shaped distribution of core peak temperature is applied at each fans.
Each Gaussian is characterized by a sigma and a peak value.
The peak of the Gaussian fan is uniformly distributed in [-range,+range] + peak.
The sigma is adjusted such as the area below the Gaussian grows linearly with the fan peak temperature.
An offset temperature can also be applied to all the cores.

Example:

[gmt_m1_thermal_influence-functions -m 5 --temp-dist fan-uniform --temp-dist-args 25e-2 60 0 0](https://github.com/rconan/gmt_m1_thermal_influence-functions/tree/main/tests/fan-uniform_25e-2_60_0_0)

### Actuator-Uniform
A Gaussian-shaped distribution of core peak temperature is applied at each actuators.
Each Gaussian is characterized by a sigma and a peak value.
The peak of the Gaussian actuator is uniformly distributed in [-range,+range] + peak.
The sigma is adjusted such as the area below the Gaussian grows linearly with the actuator peak temperature.
An offset temperature can also be applied to all the cores.

Example:

[gmt_m1_thermal_influence-functions -m 5 --temp-dist actuator-uniform --temp-dist-args 10e-2 10 0 0](https://github.com/rconan/gmt_m1_thermal_influence-functions/tree/main/tests/actuator-uniform_10e-2_10_0_0)
