# Copyright (c) OpenMMLab. All rights reserved.
from .loops import TeacherStudentValLoop
from .loops_adv import AdvTrainLoop

__all__ = ['TeacherStudentValLoop', 'AdvTrainLoop']
