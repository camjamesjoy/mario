import setuptools
import os.path
import sys

package_name = "mario"
root_dir = os.path.dirname(os.path.abspath(__file__))
req_file = os.path.join(root_dir, "requirements.txt")
package_dir = os.path.join(root_dir, package_name)

setuptools.setup(
    name=package_name,
    author="Cameron Joy",
    author_email="camjamesjoy@gmail.com",
    description="train a neural network to play mario",
    python_requires=">=3.8",
    packages=setuptools.find_packages(),
    include_package_data=True,
)
