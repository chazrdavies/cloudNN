This is a collection of hand-labeled masks of clouds, cloud-shadow, water,
and snow/ice in Landsat 8 subscenes created by M. Joseph Hughes and Rakesh
Sonanki while working with Robert Kennedy at Oregon State University.
The canonical source for this data is:
    http://emapr.ceoas.oregonstate.edu/sparcs

These masks are globally distributed across multiple ecotypes, land covers,
and cloud conditions, and are suitable for training or evaluating algorithms
that distinguish clear-sky views from obstructed views in Landsat 8 iamgery.
Additional information can be found in the following reference. The authors
request it be cited in any publications making use of this data:  

Hughes, M. Joseph and Kennedy, Robert E. (2019). "High Quality Cloud Masking
of Landsat 8 Imagery Using Convolutional Neural Networks".  Remote Sensing
(2019) Vol 11(21), 2591. MDPI. doi: 10.3390/rs11212591
https://www.mdpi.com/2072-4292/11/21/2591


This zip file contains, for each of 80 scenes:

*_labels.tif: a coded cloud, shadow, water, ice mask (see below for codes).
*_data.tif:   a 10-band geotiff of original USGS data, with bands in order.
*_mtl.txt:    the original metadata file for the entire scene
                (the extents have not been changed for the subscenes)
*_photo.png:  the image sent to the interpreter
*_qmask.tif:  the original USGS quality mask for the subscene
*_c1bqa.tif:  the Collection 1 quality mask for the subscene. More
                information about the differences between these two quality
                masks can be found at: QA Tools Userguide:
                https://landsat.usgs.gov/sites/default/files/documents/landsat_QA_tools_userguide.pdf
                Note that LC81130672013142LGN01 was not included in
                Collection 1 and does not have this file.


 Mask Code	Feature	                    Mapped Color
    0	    Cloud Shadow                Black
    1	    Cloud Shadow over Water     Dark Blue
    2	    Water                       Blue
    3	    Ice/Snow                    Cyan
    4	    Land                        Grey
    5	    Clouds                      White
    6	    Flooded                     Gold

