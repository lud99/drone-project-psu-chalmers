from setuptools import find_packages, setup

package_name = 'communication_software'

setup(
    name=package_name,
    version='2025.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='arvidboisen',
    maintainer_email='contact@arvidboisen.com',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'run_main = communication_software.main:main',
            'ros_test = communication_software.ROS:main'
        ],
    },
)
