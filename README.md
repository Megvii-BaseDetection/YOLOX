<div align="center"><img src="assets/logo.png" width="350"></div>
<br>

`pixeltable-yolox` is a lightweight, Apache-licensed object detection library built on pytorch. It is a fork of the
[MegVii YOLOX package](https://github.com/Megvii-BaseDetection/YOLOX) originally authored by Zheng Ge et al,
modernized for recent versions of Python and refactored to be more easily usable as a Python library.

## Background

The original YOLOX implementation, while powerful, has been updated only sporadically since 2022 and now faces compatibility issues with current Python environments, dependencies, and platforms like Google Colab. This fork aims to provide a reliable, up-to-date, and easy-to-use version of YOLOX that maintains its Apache license, ensuring it remains accessible for academic and commercial use.

## Status

`pixeltable-yolox` is a work in progress; we are pursuing the following changes from the base YOLOX package:

- `pip install`able with all versions of Python (3.9+)
- Maintained with regular updates
- Refactored to be usable for inference with minimal boilerplate code on the user end
- Refactored CLI for training and evaluation
- Improved test coverage with maintained CI
- Typed for use with `mypy`

We expect to have an initial release with the above changes by March 31, 2025.

## Scope

Further improvements such as model enhancements, training tooling beyond what already exists, etc., are not planned; our goal is to take the existing feature set and make it more easily usable. However, we are happy to shepherd community contributions in those areas and provide engineering infrastructure such as CI and regular releases. We intend to publish a contributors’ guide once the initial release is available.

Thanks for your interest! For any questions or feedback, please contact us at `contact@pixeltable.com`.

## Who are we and why are we doing this?

Pixeltable, Inc. is a venture-backed AI infrastructure startup. Our core product is [pixeltable](https://github.com/pixeltable/pixeltable), a database and orchestration system purpose-built for multimodal AI workloads.

Pixeltable integrates with numerous AI services and open source technologies. In the course of integrating with YOLOX, it became clear that there is a strong need in the vision community for a lightweight object detection library with an untainted open source license. It also became clear that while YOLOX is an excellent foundation, it would benefit greatly from code modernization and more regular updates.

We chose to build upon YOLOX both to simplify our own integration, and also to give something back to the community that will (hopefully) prove useful. The Pixeltable team has decades of collective experience in open source development. Our backgrounds include companies such as Google, Cloudera, Twitter, Amazon, and Airbnb, that have a strong commitment to open source development and collaboration.

## Contributing

We welcome contributions from the community! If you're interested in helping maintain and improve `pixeltable-yolox`, please stay tuned for our contributor's guide which will be published after the initial release.

## In memory of Dr. Jian Sun

Without the guidance of [Dr. Jian Sun](https://scholar.google.com/citations?user=ALVSZAYAAAAJ), YOLOX would not have been released and open sourced to the community.
The passing away of Dr. Sun is a huge loss to the Computer Vision field. We add this section here to express our remembrance and condolences to Dr. Sun.
It is hoped that every AI practitioner in the world will stick to the belief of "continuous innovation to expand cognitive boundaries, and extraordinary technology to achieve product value" and move forward all the way.

<div align="center"><img src="assets/sunjian.png" width="200"></div>
