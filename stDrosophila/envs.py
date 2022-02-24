"""Reporting your Python environment's package versions and hardware resources based on scooby.
"""
import scooby


class Report(scooby.Report):
    def __init__(self, additional=None, ncol=3, text_width=80, sort=False):
        """Initiate a scooby.Report instance."""

        # Mandatory packages.
        core = ['anndata', 'cv2', 'dask', 'dynamo', 'harmonypy',  'intermine', 'matplotlib', 'nudged', 'numpy',
                'open3d', 'ot', 'pandas', 'PVGeo', 'pyacvd', 'pyvista', 'scanpy', 'scooby', 'seaborn', 'skimage',
                'sklearn', 'spateo', 'squidpy', 'stDrosophila', 'torch'
                ]

        # Optional packages.
        optional = ['imageio-ffmpeg', 'ipyvtklink', 'jupyter', 'leidenalg', 'louvain', 'pythreejs',
                    'scanorama', 'SpaGCN', 'stereo', 'typing_extensions']

        scooby.Report.__init__(self, additional=additional, core=core,
                               optional=optional, ncol=ncol,
                               text_width=text_width, sort=sort)
