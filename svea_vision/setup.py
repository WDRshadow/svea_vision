from setuptools import setup
from glob import glob
import os

package_name = 'svea_vision'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name, 'sort'],
    package_dir={'': 'src'},
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Kaj Munhoz Arfvidsson',
    maintainer_email='kajarf@kth.se',
    description='The svea vision package',
    license='MIT',
    entry_points={
        'console_scripts': [
            'static_image_publisher = svea_vision.static_image_publisher:main',
        ],
    },
)
