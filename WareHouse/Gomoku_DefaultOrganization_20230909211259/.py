def save_load():
 filename = askopenfile('Save file', mode='w')
 try:
 f = open(filename, 'rb')
 data = serializer.dumps(board)
 f.close()
 except Exception as e:
 print('Error saving: {}'.format(str(e))
 finally:
 return filename
def load_game():
 filename = askopenfile('Load file')
 try:
 f = open(filename, 'rb')
 board = deserializer.loads(f)
 f.readline()
 f.close()
 except Exception as e:
 print('Error loading: {}'.format(str(e))
 finally:
 return board
serializer = pickle.dump(board)
deserializer = pickle.loads
pickler = pickle.Pickler(protocol=-1)