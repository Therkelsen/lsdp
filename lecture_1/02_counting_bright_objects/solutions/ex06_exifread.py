import exifread
from xml.dom.minidom import parseString

def get_gimbal_orientation(filename):
    yaw = None
    pitch = None
    roll = None

    # Open the file and extract exif information.
    f = open(filename, 'rb')
    # The debug = True option is needed to search for xmp information
    # in the .jgp file. 
    tags = exifread.process_file(f, debug=True)

    if "Image ApplicationNotes" in tags.keys():
        # Extract the xmp values and put them in a dictionary.
        dom = parseString(tags["Image ApplicationNotes"].printable)
        temp = dom.getElementsByTagName("rdf:Description")[0].attributes.items()
        attrs = dict(temp)
        # print(attrs)

        # Extract the needed information from the dictionary.
        yaw = float(attrs['drone-dji:GimbalYawDegree'])
        pitch = float(attrs['drone-dji:GimbalPitchDegree'])
        roll = float(attrs['drone-dji:GimbalRollDegree'])
    else:
        raise Exception("Could not find gimbal orientation information")

    return (yaw, pitch, roll)

print(get_gimbal_orientation('../input/DJI_0221.JPG'))
print(get_gimbal_orientation('../input/DJI_0222.JPG'))

