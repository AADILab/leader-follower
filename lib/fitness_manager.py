from lib.poi_manager import POIManager

class FitnessManager():
    def __init__(self, pm: POIManager) -> None:
        self.pm = pm

    def getTeamFitness(self):
        return float(self.pm.numObserved())/float(self.pm.num_pois)
