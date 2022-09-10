from ..display import cdisplay


def runbatch(batch,display):
    tasks=[]
    for i in range(1,1000):
        try: tasks.append((batch['task{}'.format(i)],batch['genprofile{}'.format(i)]))
        except: pass

    outputs=[]
    for task, genprofile in tasks:
        outputs.append(cdisplay.runtask(task,genprofile(outputs),display))

    return outputs
