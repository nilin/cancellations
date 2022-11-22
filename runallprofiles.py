from cancellations.utilities import setup
from cancellations.utilities import tracking, batchjob

from cancellations.examples import harmonicoscillator2d as m


def flattenprofiles(profiles):
    if isinstance(profiles,tracking.Profile):
        return {'':profiles}
    else:
        return {K+k:p.butwith(profilename=K+k) for K,P in profiles.items() for k,p in flattenprofiles(P).items()}

class Run(batchjob.Batchjob):

    processname='run all profiles'

    def runbatch(self):
        profiles=flattenprofiles(m.Run.getprofiles())

        if setup.debug:
            breakpoint()

        for k,profile in profiles.items():
            self.run=m.Run(profile)
            self.run.processname=profile.profilename
            self.run_subprocess(self.run,taskname=profile.profilename)

#
#    @staticmethod
#    def getdefaultprofile(**kw):
#        return batchjob.Batchjob.getdefaultprofile().butwith(tasks=['pick script','pick profile','run script'],**kw)
#
#
if __name__=='__main__':
    Run().run_as_main()
    setup.run_afterdisplayclosed()