import os
import sys
import pandas as pd

from sumolib import checkBinary  # noqa
import traci

import glob
import pandas as pd
from xml.etree import ElementTree
import numpy as np


import pandas as pd
import numpy as np
from xml.etree.ElementTree import Element, SubElement, tostring, XML
from xml.etree import ElementTree
from xml.dom import minidom
import os
import multiprocessing
from joblib import Parallel, delayed




def isInputNotNan(in_value):
    try:
        if ~np.isnan(in_value):
            return (1)
        else:
            return (0)
    except:
        return (1)

def prettify(elem):

    rough_string = ElementTree.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")



def createTLSXML(file, simulation_values, trafficLight_Input):
    top = Element('additional')
    parent_b = SubElement(top, 'tlLogic', id=str(simulation_values['tls_id'][0]), type=str(trafficLight_Input['type'].unique()[0]), programID='1', offset='0')
    children = []
    for v in list(trafficLight_Input.T.to_dict().values()):
        c = XML('''<phase duration="" state="" minDur="" maxDur="" name=""/> ''')
        if isInputNotNan(v['phaseDuration']):
            c.set('duration', str(v['phaseDuration']))
        if isInputNotNan(v['state']):
            c.set('state', str(v['state']))
        if isInputNotNan(v['minDur']):
            c.set('minDur', str(v['minDur']))
        if isInputNotNan(v['maxDur']):
            c.set('maxDur', str(v['maxDur']))
        if isInputNotNan(v['phaseDuration']):
            c.set('name', str(v['name']))
        children.append(c)
    # children = [
    #     Element('phase', duration=str(v['phaseDuration']), state=str(v['state']), minDur=str(isInputNotNan(v['minDur'])), maxDur=str(v['maxDur']), name=str(v['state']))
    #     for v in list(trafficLight_Input.T.to_dict().values())
    #     ]

    parent_b.extend(children)
    # Copy nodes to second parent

    # print(prettify(top))

    
    # os.makedirs(os.path.dirname(file), exist_ok=True)

    myfile = open(file, "w")
    myfile.write(prettify(top))



def runSingleSimulation(file = 'osm.sumocfg'):
    sumoBinary = checkBinary('sumo')
    traci.start([sumoBinary, "-c", file, '--no-warnings'])

    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()    
    traci.close()

def get_output_summary(pcr_signal, iqbalminar_signal, oldpssaifabad_signal):
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("please declare environment variable 'SUMO_HOME'")

    amber_signal = 5
    data = [['run','static',pcr_signal,'GGGGGGGGGGGGrrrrrrrrrrr','10','100', 'pcr'],
            ['run','static',amber_signal,'GGGGGYYYYYYGrrrrrrrrrrr','10','100', 'amber0'],
            ['run','static',iqbalminar_signal,'GGGGgrrrrrrGGGGGGrrrrrr','10','100', 'iqbalminar'],
            ['run','static',amber_signal,'GGGGgrrrrrrGYYYYYrrrrrr','10','100', 'amber1'],
            ['run','static',oldpssaifabad_signal,'GGGGGrrrrrrgrrrrrGGGGGG','10','100', 'oldpssaifabad'],
            ['run','static',amber_signal,'GGGGGrrrrrrGrrrryYYYYYY','10','100', 'amber2']
        ]
    trafficLight_Input = pd.DataFrame(data, columns = ['IntervalID', 'type', 'phaseDuration', 'state', 'minDur', 'maxDur', 'name']) 

    simulation_values = pd.DataFrame([['3600', 'cluster_2177311110_2422989108_289917156_308769395']], columns = ['duration', 'tls_id'])

    createTLSXML(file = 'tls.add.xml', simulation_values = simulation_values, trafficLight_Input = trafficLight_Input)

    runSingleSimulation()

    text_files = glob.glob('data'+os.sep+"*detector.xml", recursive = True)

    values = ['begin', 'end', 'id', 'nVehContrib', 'meanSpeed', 'meanTimeLoss', 'meanTravelTime', 'vehicleSum', 'vehicleSumWithin']
    output = []

    for infile in text_files:
        tree = ElementTree.parse(infile)
        root = tree.getroot()
        for att in root:
            line = {}
            for v in values:
        #         print(att.attrib.get(v))
                line[v] = att.attrib.get(v)
            output.append(line)
    
    output = pd.DataFrame(output)

    for f in ['begin', 'end', 'nVehContrib', 'meanSpeed', 'meanTimeLoss', 'meanTravelTime', 'vehicleSum', 'vehicleSumWithin']:
        output[f] = output[f].astype('float')
        
    output = output[output['vehicleSumWithin']>0]
    output['arm'] = output.apply(lambda x: x['id'].split('_')[0],axis=1)

    output_summary = output.groupby('arm')['meanTimeLoss'].mean().to_dict()

    # print(output_summary)
    return output_summary

def getSummaryFromSummaryOutputFile(infile, rmP):
    tree = ElementTree.parse(infile)
    root = tree.getroot()
    # print(root[0].keys())

    output = []
    for att in root:
        values = {}
        values['time'] = att.attrib.get("time")
        values['meanWaitingTime'] = att.attrib.get("meanWaitingTime")
        values['meanSpeed'] = att.attrib.get("meanSpeed")
        values['meanTravelTime'] = att.attrib.get("meanTravelTime")
        values['running'] = att.attrib.get("running")
        values['stopped'] = att.attrib.get("stopped")
        output.append(values)

    removal = int(len(output) * rmP / 100)
    output = output[removal:- 4* removal]

    df = pd.DataFrame(output)
    df['time'] = df.time.astype(float)
    df['meanWaitingTime'] = df.meanWaitingTime.astype(float)
    df['meanSpeed'] = df.meanSpeed.astype(float)
    df['meanTravelTime'] = df.meanTravelTime.astype(float)
    df['running'] = df.running.astype(float)
    df['stopped'] = df.stopped.astype(float)

    df_summary = df.describe()

    return (df_summary, df)


def main(signalDict):
    output_summary = get_output_summary(signalDict['pcr_signal'],signalDict['iqbalminar_signal'], signalDict['oldpssaifabad_signal'])
    output_summary['pcr_signal'] = signalDict['pcr_signal']
    output_summary['iqbalminar_signal'] = signalDict['iqbalminar_signal']
    output_summary['oldpssaifabad_signal'] = signalDict['oldpssaifabad_signal']
    df_summary, value = getSummaryFromSummaryOutputFile('data'+os.sep+'summary-output.xml',5)
    output_summary['overallMeanSpeed'] = df_summary.iloc[1]['meanSpeed']
    output_summary['overallmeanTravelTime'] = df_summary.iloc[1]['meanTravelTime']
    df = pd.DataFrame([output_summary])
    df.to_csv('output.csv', mode='a', header=True)

if __name__ == '__main__':

    ##input
    pcr_signal = 35
    iqbalminar_signal = 19
    oldpssaifabad_signal = 81

    inputs = []
    for x in range(30,40):
        pcr_signal = x
        for y in range(15,25):
            iqbalminar_signal = y
            for z in range(75,85):
                oldpssaifabad_signal = z
                signalDict = {
                    'pcr_signal' : pcr_signal,
                    'iqbalminar_signal' : iqbalminar_signal,
                    'oldpssaifabad_signal' : oldpssaifabad_signal
                }
                inputs.append(signalDict)

    ## 
    num_cores = multiprocessing.cpu_count()
    Parallel(n_jobs=num_cores)(delayed(main(value)) for value in inputs)