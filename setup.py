#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from setuptools import setup, find_packages
import subspacekmeans


def setup_package():
	metadata = dict()
	metadata['name'] = subspacekmeans.__package__
	metadata['version'] = subspacekmeans.__version__
	metadata['description'] = subspacekmeans.description_
	metadata['author'] = subspacekmeans.author_
	metadata['url'] = subspacekmeans.url_
	metadata['license'] = 'MIT'
	metadata['packages'] = find_packages()
	metadata['include_package_data'] = False
	metadata['install_requires'] = [
		'scikit-learn',
	]
	setup(**metadata)


if __name__ == "__main__":
	setup_package()
