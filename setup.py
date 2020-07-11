from setuptools import setup

description = 'Tools able to generate long term timeseries of WEC timeseries output'

setup(
        name='pyWECcast',
        version='0.2.2',
        description=description,
        py_modules=['pyWECcast'],
        package_dir={'':'pyWECcast'},
        url='https://github.com/aidanbharath/pyWECcast',
        author='Aidan Bharath',
        author_email='Aidan.Bharath@nrel.gov',
    )
