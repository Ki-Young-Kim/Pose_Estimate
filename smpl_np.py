import numpy as np
import pickle
import argparse


class SMPLModel():
  def __init__(self, model_path, k=23):
    """
    SMPL model.

    Parameter:
    ---------
    model_path: Path to the SMPL model parameters, pre-processed by
    `preprocess.py`.

    """
    with open(model_path, 'rb') as f:
      try:
        params = pickle.load(f, encoding='latin1')
      except pickle.UnpicklingError:
        params = np.load(f)

      self.J_regressor = params['J_regressor']
      self.weights = params['weights']
      self.posedirs = params['posedirs']
      self.v_template = params['v_template']
      self.shapedirs = params['shapedirs']
      self.faces = params['f']
      self.kintree_table = params['kintree_table']

    id_to_col = {
      self.kintree_table[1, i]: i for i in range(self.kintree_table.shape[1])
    }
    self.parent = {
      i: id_to_col[self.kintree_table[0, i]]
      for i in range(1, self.kintree_table.shape[1])
    }

    self.beta_sz = self.shapedirs.shape[-1]

    self.k = k
    self.pose_shape = [k + 1, 3]
    self.beta_shape = [self.beta_sz]
    self.trans_shape = [3]

    self.pose = np.zeros(self.pose_shape)
    self.beta = np.zeros(self.beta_shape)
    self.trans = np.zeros(self.trans_shape)

    self.verts = None
    self.J = None
    self.R = None

    self.G = None

    self.update()

  def set_params(self, pose=None, beta=None, trans=None, skip_verts_update=False):
    """
    Set pose, shape, and/or translation parameters of SMPL model. Verices of the
    model will be updated and returned.

    Parameters:
    ---------
    pose: Also known as 'theta', a [24,3] matrix indicating child joint rotation
    relative to parent joint. For root joint it's global orientation.
    Represented in a axis-angle format.

    beta: Parameter for model shape. A vector of shape [10]. Coefficients for
    PCA component. Only 10 components were released by MPI.

    trans: Global translation of shape [3].

    Return:
    ------
    Updated vertices.

    """
    if pose is not None:
      self.pose = pose
    if beta is not None:
      if len(beta) > self.beta_sz:
        raise ValueError("Size of supplied beta is greater than assigned beta size: {}".format(self.beta_sz))
      elif len(beta) < self.beta_sz:
        #print("Size of supplied beta is less than assigned beta size {}; it will be padded with zeroes".format(self.beta_sz))
        beta = np.pad(beta, (0, self.beta_sz - len(beta)), mode='constant')
      self.beta = beta
    if trans is not None:
      self.trans = trans
    self.update(skip_verts_update=skip_verts_update)
    return self.verts

  def update(self, skip_verts_update=False):
    """
    Called automatically when parameters are updated.

    """
    # how beta affect body shape
    v_shaped = self.shapedirs.dot(self.beta) + self.v_template
    # joints location
    self.J = self.J_regressor.dot(v_shaped)
    pose_cube = self.pose.reshape((-1, 1, 3))
    # rotation matrix for each joint
    self.R = self.rodrigues(pose_cube)
    if not skip_verts_update:
      I_cube = np.broadcast_to(
        np.expand_dims(np.eye(3), axis=0),
        (self.R.shape[0]-1, 3, 3)
      )
      lrotmin = (self.R[1:] - I_cube).ravel()
      # how pose affect body shape in zero pose
      v_posed = v_shaped + self.posedirs.dot(lrotmin)
      # world transformation of each joint
      G = np.empty((self.kintree_table.shape[1], 4, 4))
      G[0] = self.with_zeros(np.hstack((self.R[0], self.J[0, :].reshape([3, 1]))))
      for i in range(1, self.kintree_table.shape[1]):
        G[i] = G[self.parent[i]].dot(
          self.with_zeros(
            np.hstack(
              [self.R[i],((self.J[i, :]-self.J[self.parent[i],:]).reshape([3,1]))]
            )
          )
        )
      G = G - self.pack(
        np.matmul(
          G,
          np.hstack([self.J, np.zeros([self.k + 1, 1])]).reshape([self.k + 1, 4, 1])
          )
        )
      self.G = G
      # transformation of each vertex
      T = np.tensordot(self.weights, G, axes=[[1], [0]])
      rest_shape_h = np.hstack((v_posed, np.ones([v_posed.shape[0], 1])))
      v = np.matmul(T, rest_shape_h.reshape([-1, 4, 1])).reshape([-1, 4])[:, :3]
      self.verts = v + self.trans.reshape([1, 3])

  def rodrigues(self, r):
    """
    Rodrigues' rotation formula that turns axis-angle vector into rotation
    matrix in a batch-ed manner.

    Parameter:
    ----------
    r: Axis-angle rotation vector of shape [batch_size, 1, 3].

    Return:
    -------
    Rotation matrix of shape [batch_size, 3, 3].

    """
    theta = np.linalg.norm(r, axis=(1, 2), keepdims=True)
    # avoid zero divide
    theta = np.maximum(theta, np.finfo(r.dtype).eps)
    r_hat = r / theta
    cos = np.cos(theta)
    z_stick = np.zeros(theta.shape[0])
    m = np.dstack([
      z_stick, -r_hat[:, 0, 2], r_hat[:, 0, 1],
      r_hat[:, 0, 2], z_stick, -r_hat[:, 0, 0],
      -r_hat[:, 0, 1], r_hat[:, 0, 0], z_stick]
    ).reshape([-1, 3, 3])
    i_cube = np.broadcast_to(
      np.expand_dims(np.eye(3), axis=0),
      [theta.shape[0], 3, 3]
    )
    A = np.transpose(r_hat, axes=[0, 2, 1])
    B = r_hat
    dot = np.matmul(A, B)
    R = cos * i_cube + (1 - cos) * dot + np.sin(theta) * m
    return R

  def with_zeros(self, x):
    """
    Append a [0, 0, 0, 1] vector to a [3, 4] matrix.

    Parameter:
    ---------
    x: Matrix to be appended.

    Return:
    ------
    Matrix after appending of shape [4,4]

    """
    return np.vstack((x, np.array([[0.0, 0.0, 0.0, 1.0]])))

  def pack(self, x):
    """
    Append zero matrices of shape [4, 3] to vectors of [4, 1] shape in a batched
    manner.

    Parameter:
    ----------
    x: Matrices to be appended of shape [batch_size, 4, 1]

    Return:
    ------
    Matrix of shape [batch_size, 4, 4] after appending.

    """
    return np.dstack((np.zeros((x.shape[0], 4, 3)), x))

  def save_to_obj(self, path):
    """
    Save the SMPL model into .obj file.

    Parameter:
    ---------
    path: Path to save.

    """
    with open(path, 'w') as fp:
      for v in self.verts:
        fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
      for f in self.faces + 1:
        fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))

  def pose_joints(self):

      #self.R[0] = np.eye(self.R[0].shape[0])

      rel_joints = self.J.copy()
      for i in range(1, self.k + 1):
          rel_joints[i] -= self.J[self.parent[i]]

      transform_mats = np.array([np.vstack(
          [np.hstack([self.R[i], rel_joints[i].reshape((3, 1))]), [[0, 0, 0, 1]]])
          for i in range(self.k + 1)])
  
      transform_chain = [transform_mats[0]]
      for i in range(1, self.k + 1):
          cur_rest = transform_chain[self.parent[i]] @ transform_mats[i]
          transform_chain.append(cur_rest)
  
      posed_joints = np.array(transform_chain)[:, :3, 3]

      return posed_joints


joint_labels = {
        0: "Pelvis",
        1: "L_Hip",
        2: "R_Hip",
        3: "Spine1",
        4: "L_Knee",
        5: "R_Knee",
        6: "Spine2",
        7: "L_Ankle",
        8: "R_Ankle",
        9: "Spine3",
        10: "L_Foot",
        11: "R_Foot",
        12: "Neck",
        13: "L_Collar",
        14: "R_Collar",
        15: "Head",
        16: "L_Shoulder",
        17: "R_Shoulder",
        18: "L_Elbow",
        19: "R_Elbow",
        20: "L_Wrist",
        21: "R_Wrist",
        22: "L_Hand",
        23: "R_Hand",
}


if __name__ == '__main__':
  ap = argparse.ArgumentParser()
  ap.add_argument('action', choices=['test_model', 'preprocess'])
  ap.add_argument('argv', nargs='*')
  
  aa = ap.parse_args()

  if aa.action == 'test_model':
    smpl = SMPLModel('./model.pkl')
    np.random.seed(9608)
    pose = (np.random.rand(*smpl.pose_shape) - 0.5) * 0.4
    beta = (np.random.rand(*smpl.beta_shape) - 0.5) * 0.06
    trans = np.zeros(smpl.trans_shape)
    smpl.set_params(beta=beta, pose=pose, trans=trans)
    smpl.save_to_obj('./smpl_np.obj')
  elif aa.action == 'preprocess':
    output_path = './model.pkl'
    if len(aa.argv) < 2:
      print('Error: expected source model path.')
      exit(-1)
    src_path = aa.argv[1]
    with open(src_path, 'rb') as f:
      src_data = pickle.load(f, encoding="latin1")
    model = {
      'J_regressor': src_data['J_regressor'],
      'weights': np.array(src_data['weights']),
      'posedirs': np.array(src_data['posedirs']),
      'v_template': np.array(src_data['v_template']),
      'shapedirs': np.array(src_data['shapedirs']),
      'f': np.array(src_data['f']),
      'kintree_table': src_data['kintree_table']
    }
    if 'cocoplus_regressor' in src_data.keys():
      model['joint_regressor'] = src_data['cocoplus_regressor']
    with open(output_path, 'wb') as f:
      pickle.dump(model, f)
  else:
    exit(1)
