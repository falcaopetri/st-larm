import numpy as np
from trajminer import TrajectoryData
from trajminer.preprocessing import TrajectorySegmenter

attributes = ['poi', 'hour', 'rating']
data = np.array([[['Bakery', 8, 8.6], ['Work', 9, 8.9],
              ['Restaurant', 12, 7.7], ['Bank', 12, 5.6],
              ['Work', 13, 8.9], ['Home', 19, 0.0]],
             [['Home', 8, 0.0], ['Mall', 10, 9.3],
              ['Home', 19, 0.0], ['Pub', 21, 9.5]]])
tids = np.array([20, 24])
labels = np.array([1, 2])
traj = TrajectoryData(attributes=attributes,
                      data=data,
                      tids=tids,
                      labels=labels)

def _get_correspondings(lst1, lst2, lst1_value):
    lst1 = np.array(lst1)
    lst2 = np.array(lst2)
    return lst2[lst1 == lst1_value]

class TestTrajectoryData:
        
    def test_attributes(self):
        assert np.array_equal(attributes, traj.get_attributes())
        
    def test_trajectories(self):
        assert np.array_equal(data, traj.get_trajectories())
        
    def test_tids(self):
        assert np.array_equal(tids, traj.get_tids())  
    
    def test_tids_by_label1(self):
        target_label = 1
        expected_tids = _get_correspondings(labels, 
                                            tids, 
                                            target_label)
        assert np.array_equal(expected_tids,
                              traj.get_tids(label=target_label))
    
    def test_tids_by_label2(self):
        target_label = 2
        expected_tids = _get_correspondings(labels, 
                                            tids, 
                                            target_label)
        assert np.array_equal(expected_tids,
                              traj.get_tids(label=target_label))
    
    def test_label_by_tid20(self):
        target_tid = 20
        expected_label = _get_correspondings(tids, 
                                             labels, 
                                             target_tid)[0]
        assert expected_label == traj.get_label(tid=target_tid)
        
    def test_label_by_tid24(self):
        target_tid = 24
        expected_label = _get_correspondings(tids, 
                                             labels, 
                                             target_tid)[0]
        assert expected_label == traj.get_label(tid=target_tid)
        
    def test_labels(self):
        assert np.array_equal(labels, traj.get_labels())
        
    def test_labels_unique(self):
        # TODO
        assert np.array_equal(sorted(set(labels)), 
                              traj.get_labels(unique=True))
    
    def test_trajectory_by_tid20(self):
        target_tid = 20
        expected_data = _get_correspondings(tids, 
                                            data, 
                                            target_tid)[0]
        assert np.array_equal(expected_data,
                              traj.get_trajectory(tid=target_tid))
        
    def test_trajectory_by_tid24(self):
        target_tid = 20
        expected_data = _get_correspondings(tids, 
                                            data, 
                                            target_tid)[0]
        assert np.array_equal(expected_data,
                              traj.get_trajectory(tid=target_tid))
        
    def test_trajectories(self):
        assert np.array_equal(data, traj.get_trajectories())
        
    def test_trajectories_by_label1(self):
        target_label = 1
        expected_data = _get_correspondings(labels, 
                                            data, 
                                            target_label)
        assert np.array_equal(expected_data,
                              traj.get_trajectories(label=target_label))
        
    def test_trajectories_by_label2(self):
        target_label = 2
        expected_data = _get_correspondings(labels, 
                                            data, 
                                            target_label)
        assert np.array_equal(expected_data,
                              traj.get_trajectories(label=target_label))
        
    def test_length(self):
        assert len(tids) == traj.length()
    