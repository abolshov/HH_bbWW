from abc import ABC, abstractmethod


class ProducerBase(ABC):
    def __init__(self, cfg):
        self.cfg = cfg 

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def run(self, dfw):
        pass 

    @abstractmethod
    def format(self):
        pass


class HMEProducer(ProducerBase):
    def __init__(self, cfg):
        self.initialize()

    def initialize(self):
        pass

    def run(self, dfw):
        pass