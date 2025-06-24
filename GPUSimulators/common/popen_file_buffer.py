import tempfile


class PopenFileBuffer(object):
    """
    Simple class for holding a set of temp files
    for communicating with a subprocess
    """

    def __init__(self):
        self.stdout = tempfile.TemporaryFile(mode='w+t')
        self.stderr = tempfile.TemporaryFile(mode='w+t')

    def __del__(self):
        self.stdout.close()
        self.stderr.close()

    def read(self):
        self.stdout.seek(0)
        cout = self.stdout.read()
        self.stdout.seek(0, 2)

        self.stderr.seek(0)
        cerr = self.stderr.read()
        self.stderr.seek(0, 2)

        return cout, cerr