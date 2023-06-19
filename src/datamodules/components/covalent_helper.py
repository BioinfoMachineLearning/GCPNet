# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for GCPNet (https://github.com/BioinfoMachineLearning/GCPNet):
# -------------------------------------------------------------------------------------------------------------------------------------

import numpy as np
import pandas as pd

from biopandas.pdb import PandasPdb
from scipy.spatial.distance import pdist, squareform
from typing import Any, Callable, Dict, List, Optional

from torchtyping import patch_typeguard
from typeguard import typechecked

patch_typeguard()  # use before @typechecked


# Atom classes based on Heyrovska, Raji covalent radii paper.
DEFAULT_BOND_STATE: Dict[str, str] = {
    "N": "Nsb",
    "CA": "Csb",
    "C": "Cdb",
    "O": "Odb",
    "OXT": "Osb",
    "CB": "Csb",
    "H": "Hsb",
    # Not sure about these - assuming they're all standard Hydrogen. Won't make much difference given
    # the tolerance is larger than Hs covalent radius
    "HG1": "Hsb",
    "HE": "Hsb",
    "1HH1": "Hsb",
    "1HH2": "Hsb",
    "2HH1": "Hsb",
    "2HH2": "Hsb",
    "HG": "Hsb",
    "HH": "Hsb",
    "1HD2": "Hsb",
    "2HD2": "Hsb",
    "HZ1": "Hsb",
    "HZ2": "Hsb",
    "HZ3": "Hsb",
}

RESIDUE_ATOM_BOND_STATE: Dict[str, Dict[str, str]] = {
    "XXX": {
        "N": "Nsb",
        "CA": "Csb",
        "C": "Cdb",
        "O": "Odb",
        "OXT": "Osb",
        "CB": "Csb",
        # "H": "Hsb",
    },
    "VAL": {"CG1": "Csb", "CG2": "Csb"},
    "LEU": {"CG": "Csb", "CD1": "Csb", "CD2": "Csb"},
    "ILE": {"CG1": "Csb", "CG2": "Csb", "CD1": "Csb"},
    "MET": {"CG": "Csb", "SD": "Ssb", "CE": "Csb"},
    "PHE": {
        "CG": "Cdb",
        "CD1": "Cres",
        "CD2": "Cres",
        "CE1": "Cdb",
        "CE2": "Cdb",
        "CZ": "Cres",
    },
    "PRO": {"CG": "Csb", "CD": "Csb"},
    "SER": {"OG": "Osb"},
    "THR": {"OG1": "Osb", "CG2": "Csb"},
    "CYS": {"SG": "Ssb"},
    "ASN": {"CG": "Csb", "OD1": "Odb", "ND2": "Ndb"},
    "GLN": {"CG": "Csb", "CD": "Csb", "OE1": "Odb", "NE2": "Ndb"},
    "TYR": {
        "CG": "Cdb",
        "CD1": "Cres",
        "CD2": "Cres",
        "CE1": "Cdb",
        "CE2": "Cdb",
        "CZ": "Cres",
        "OH": "Osb",
    },
    "TRP": {
        "CG": "Cdb",
        "CD1": "Cdb",
        "CD2": "Cres",
        "NE1": "Nsb",
        "CE2": "Cdb",
        "CE3": "Cdb",
        "CZ2": "Cres",
        "CZ3": "Cres",
        "CH2": "Cdb",
    },
    "ASP": {"CG": "Csb", "OD1": "Ores", "OD2": "Ores"},
    "GLU": {"CG": "Csb", "CD": "Csb", "OE1": "Ores", "OE2": "Ores"},
    "HIS": {
        "CG": "Cdb",
        "CD2": "Cdb",
        "ND1": "Nsb",
        "CE1": "Cdb",
        "NE2": "Ndb",
    },
    "LYS": {"CG": "Csb", "CD": "Csb", "CE": "Csb", "NZ": "Nsb"},
    "ARG": {
        "CG": "Csb",
        "CD": "Csb",
        "NE": "Nsb",
        "CZ": "Cdb",
        "NH1": "Nres",
        "NH2": "Nres",
    },
}

# Covalent radii for OpenSCAD output.

# Covalent radii from Heyrovska, Raji : 'Atomic Structures of all the Twenty
# Essential Amino Acids and a Tripeptide, with Bond Lengths as Sums of Atomic
# Covalent Radii' <https://arxiv.org/pdf/0804.2488.pdf>
# Adding Ores between Osb and Odb for Asp and Glu, Nres between Nsb and Ndb
# for Arg, as PDB does not specify

COVALENT_RADII: Dict[str, float] = {
    "Csb": 0.77,
    "Cres": 0.72,
    "Cdb": 0.67,
    "Osb": 0.67,
    "Ores": 0.635,
    "Odb": 0.60,
    "Nsb": 0.70,
    "Nres": 0.66,
    "Ndb": 0.62,
    "Hsb": 0.37,
    "Ssb": 1.04,
}


# These are the covalent distance tolerances
TOLERANCE = 0.56  # e.g., [0.4, 0.45, 0.56]


@typechecked
def filter_dataframe(
    dataframe: pd.DataFrame,
    by_column: str,
    list_of_values: List[Any],
    boolean: bool,
) -> pd.DataFrame:
    """
    Filter function for dataframe.
    Filters the [dataframe] such that the [by_column] values have to be
    in the [list_of_values] list if boolean == True, or not in the list
    if boolean == False

    :param dataframe: pd.DataFrame to filter
    :type dataframe: pd.DataFrame
    :param by_column: str denoting by_column of dataframe to filter
    :type by_column: str
    :param list_of_values: List of values to filter with
    :type list_of_values: List[Any]
    :param boolean: indicates whether to keep or exclude matching list_of_values. True -> in list, false -> not in list
    :type boolean: bool
    :returns: Filtered dataframe
    :rtype: pd.DataFrame
    """
    df = dataframe.copy()
    df = df[df[by_column].isin(list_of_values) == boolean]
    df.reset_index(inplace=True, drop=True)

    return df


@typechecked
def deprotonate_structure(df: pd.DataFrame) -> pd.DataFrame:
    """Remove protons from PDB dataframe.

    :param df: Atomic dataframe.
    :type df: pd.DataFrame
    :returns: Atomic dataframe with all atom_name == "H" removed.
    :rtype: pd.DataFrame
    """
    return filter_dataframe(
        df, by_column="atom_name", list_of_values=['H', 'H2', 'H3', 'HA', 'HB2', 'HB3', 'HD1', 'HD2', 'HE1', 'HE2', 'HZ', 'HG2', 'HG3', 'HE21', 'HE22', 'HG', 'HD11', 'HD12', 'HD13', 'HD21', 'HD22', 'HD23', 'HB', 'HG21', 'HG22', 'HG23', 'HG1', 'HD3', 'HH', 'HG11', 'HG12', 'HG13', 'HE3', 'HZ1', 'HZ2', 'HZ3', 'HH2', 'HE', 'HH11', 'HH12', 'HH21', 'HH22', 'HA2', 'HA3', 'HB1'], boolean=False
    )


@typechecked
def convert_structure_to_centroids(df: pd.DataFrame) -> pd.DataFrame:
    """Overwrite existing (x, y, z) coordinates with centroids of the amino acids.

    :param df: Pandas Dataframe config protein structure to convert into a dataframe of centroid positions
    :type df: pd.DataFrame
    :return: pd.DataFrame with atoms/residues positions converted into centroid positions
    :rtype: pd.DataFrame
    """

    centroids = calculate_centroid_positions(df)
    df = df.loc[df["atom_name"] == "CA"].reset_index(drop=True)
    df["x_coord"] = centroids["x_coord"]
    df["y_coord"] = centroids["y_coord"]
    df["z_coord"] = centroids["z_coord"]

    return df


@typechecked
def calculate_centroid_positions(atoms: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Calculates position of sidechain centroids

    :param atoms: ATOM df of protein structure
    :type atoms: pd.DataFrame
    :param verbose: bool controlling verbosity
    :type verbose: bool
    :return: centroids (df)
    :rtype: pd.DataFrame
    """
    centroids = (
        atoms.groupby("residue_number")
        .mean()[["x_coord", "y_coord", "z_coord"]]
        .reset_index()
    )
    if verbose:
        print(f"Calculated {len(centroids)} centroid nodes")

    return centroids


@typechecked
def remove_insertions(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function removes insertions from PDB dataframes

    :param df: Protein Structure dataframe to remove insertions from
    :type df: pd.DataFrame
    :return: Protein structure dataframe with insertions removed
    :rtype: pd.DataFrame
    """
    """Remove insertions from structure."""
    return filter_dataframe(
        df, by_column="alt_loc", list_of_values=["", "A"], boolean=True
    )


@typechecked
def read_pdb_file_to_pdb(
    pdb_path: Optional[str] = None,
    pdb_code: Optional[str] = None,
    verbose: bool = False,
    granularity: str = "atom"
) -> PandasPdb:
    """
    Reads PDB file to PandasPDB object.

    Returns `atomic_df`, which is a dataframe enumerating all atoms and their cartesian coordinates in 3D space. Also
        contains associated metadata.

    :param pdb_path: path to PDB file. Defaults to None.
    :type pdb_path: str, optional
    :param pdb_code: 4-character PDB accession. Defaults to None.
    :type pdb_code: str, optional
    :param verbose: print dataframe?
    :type verbose: bool
    :param granularity: Specifies granularity of dataframe. See graphein.protein.config.ProteinGraphConfig for further
        details.
    :type granularity: str
    :returns: PandasPdb containing protein structure
    :rtype: PandasPdb
    """
    if pdb_code is None and pdb_path is None:
        raise NameError("One of pdb_code or pdb_path must be specified!")

    atomic_df = (
        PandasPdb().read_pdb(pdb_path)
        if pdb_path is not None
        else PandasPdb().fetch_pdb(pdb_code)
    )

    # assign Node IDs to dataframes
    atomic_df.df["ATOM"]["node_id"] = (
        atomic_df.df["ATOM"]["chain_id"].apply(str)
        + ":"
        + atomic_df.df["ATOM"]["residue_name"]
        + ":"
        + atomic_df.df["ATOM"]["residue_number"].apply(str)
    )
    if granularity == "atom":
        atomic_df.df["ATOM"]["node_id"] = (
            atomic_df.df["ATOM"]["node_id"]
            + ":"
            + atomic_df.df["ATOM"]["atom_name"]
        )
    if verbose:
        print(atomic_df.df['ATOM'])

    return atomic_df


@typechecked
def select_chains(protein_df: pd.DataFrame, chain_selection: str) -> pd.DataFrame:
    """
    Extracts relevant chains from protein_df

    :param protein_df: pandas dataframe of PDB subsetted to relevant atoms (CA, CB)
    :type protein_df: pd.DataFrame
    :param chain_selection: Specifies chains that should be extracted from the larger complexed structure
    :type chain_selection: str
    :return Protein structure dataframe containing only entries in the chain selection
    :rtype: pd.DataFrame
    """
    if chain_selection != "all":
        protein_df = filter_dataframe(
            protein_df,
            by_column="chain_id",
            list_of_values=list(chain_selection),
            boolean=True,
        )

    return protein_df


@typechecked
def compute_distmat(pdb_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute pairwise euclidean distances between every atom.

    Design choice: passed in a DataFrame to enable easier testing on
    dummy data.

    :param pdb_df: pd.Dataframe containing protein structure. Must contain columns ["x_coord", "y_coord", "z_coord"]
    :type pdb_df: pd.DataFrame
    :return: pd.Dataframe of euclidean distance matrix
    :rtype: pd.DataFrame
    """
    eucl_dists = pdist(
        pdb_df[["x_coord", "y_coord", "z_coord"]], metric="euclidean"
    )
    eucl_dists = pd.DataFrame(squareform(eucl_dists))
    eucl_dists.index = pdb_df.index
    eucl_dists.columns = pdb_df.index

    return eucl_dists


@typechecked
def assign_bond_states_to_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a PandasPDB atom dataframe and assigns bond states to each atom based on:
    Atomic Structures of all the Twenty Essential Amino Acids and a Tripeptide, with Bond Lengths as Sums of Atomic Covalent Radii
    Heyrovska, 2008

    :param df: Pandas PDB dataframe
    :type df: pd.DataFrame
    :return: Dataframe with added atom_bond_state column
    :rtype: pd.DataFrame
    """

    # map atoms to their standard bond states
    naive_bond_states = pd.Series(df["atom_name"].map(DEFAULT_BOND_STATE))

    # create series of bond states for the non-standard states
    ss = (
        pd.DataFrame(RESIDUE_ATOM_BOND_STATE)
        .unstack()
        .rename_axis(("residue_name", "atom_name"))
        .rename("atom_bond_state")
    )

    # map non-standard states to the dataframe based on the residue and atom name
    df = df.join(ss, on=["residue_name", "atom_name"])

    # fill the NaNs with the standard states
    df = df.fillna(value={"atom_bond_state": naive_bond_states})

    return df


@typechecked
def assign_covalent_radii_to_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assigns covalent radius to each atom based on its bond state. Using Values from :
    Atomic Structures of all the Twenty Essential Amino Acids and a Tripeptide, with Bond Lengths as Sums of Atomic Covalent Radii
    Heyrovska, 2008

    :param df: Pandas PDB dataframe with a bond_states_column
    :type df: pd.DataFrame
    :return: Pandas PDB dataframe with added covalent_radius column
    :rtype: pd.DataFrame
    """
    # assign covalent radius to each atom
    df["covalent_radius"] = df["atom_bond_state"].map(COVALENT_RADII)

    return df


@typechecked
def filter_hetatms(df: pd.DataFrame, keep_hets: List[str]) -> List[pd.DataFrame]:
    """Return hetatms of interest.
    :param df: Protein Structure dataframe to filter hetatoms from.
    :type df: pd.DataFrame
    :param keep_hets: List of hetero atom names to keep.
    :type keep_hets: List[str]
    :returns: Protein structure dataframe with heteroatoms removed
    :rtype: pd.DataFrame
    """
    return [df.loc[df["residue_name"] == hetatm] for hetatm in keep_hets]


@typechecked
def subset_structure_to_atom_type(df: pd.DataFrame, granularity: str) -> pd.DataFrame:
    """
    Return a subset of atomic dataframe that contains only certain atom names.
    :param df: Protein Structure dataframe to subset.
    :type df: pd.DataFrame
    :returns: Subset protein structure dataframe.
    :rtype: pd.DataFrame
    """
    return filter_dataframe(
        df, by_column="atom_name", list_of_values=[granularity], boolean=True
    )


@typechecked
def process_pdb(
    protein_pdb: PandasPdb,
    atom_df_processing_funcs: Optional[List[Callable]] = None,
    hetatom_df_processing_funcs: Optional[List[Callable]] = None,
    granularity: str = "atoms",
    chain_selection: str = "all",
    insertions: bool = False,
    deprotonate: bool = True,
    keep_hets: List[str] = []
) -> pd.DataFrame:
    """
    Process ATOM and HETATM dataframes to produce singular dataframe used for graph construction

    :param protein_pdb: PandasPdb to process.
        Should be the object returned from `read_pdb_to_dataframe`.
    :type protein_pdb: PandasPdb
    :param atom_df_processing_funcs: List of functions to process dataframe. These must take in a dataframe and return a
        dataframe. Defaults to None.
    :type atom_df_processing_funcs: List[Callable], optional
    :param hetatom_df_processing_funcs: List of functions to process dataframe. These must take in a dataframe and return a dataframe
    :type hetatom_df_processing_funcs: List[Callable], optional
    :param granularity: The level of granularity for the graph.
        This determines the node definition.
        Acceptable values include:
        - "centroids"
        - "atoms"
        - any of the atom_names in the PDB file (e.g. "CA", "CB", "OG", etc.)
    :type granularity: str
    :param insertions: Whether or not to keep insertions.
    :param insertions: bool
    :param deprotonate: Whether or not to remove hydrogen atoms (i.e. deprotonation).
    :type deprotonate: bool
    :param keep_hets: Hetatoms to keep. Defaults to an empty list.
        To keep a hetatom, pass it inside a list of hetatom names to keep.
    :type keep_hets: List[str]
    :param chain_selection: Which protein chain to select. Defaults to "all". Eg can use "ACF"
        to select 3 chains (A, C & F :)
    :type chain_selection: str
    :return: A DataFrame that can be consumed by
        other graph construction functions.
    :rtype: pd.DataFrame
    """
    atoms = protein_pdb.df["ATOM"]
    hetatms = protein_pdb.df["HETATM"]

    # Note: This block enables processing via a list of supplied functions operating on the atom and hetatom dataframes.
    # If these are provided, the DataFrame returned will be computed only from these and the default workflow
    # below this block will not execute.
    if atom_df_processing_funcs is not None:
        for func in atom_df_processing_funcs:
            atoms = func(atoms)
        if hetatom_df_processing_funcs is None:
            return atoms

    if hetatom_df_processing_funcs is not None:
        for func in hetatom_df_processing_funcs:
            hetatms = func(hetatms)
        return pd.concat([atoms, hetatms])

    # deprotonate structure by removing H atoms
    if deprotonate:
        atoms = deprotonate_structure(atoms)
        atoms = atoms[(atoms['atom_name'] != 'OXT')]

    # restrict DataFrame to desired granularity
    if granularity == "atom":
        pass
    elif granularity == "centroids":
        atoms = convert_structure_to_centroids(atoms)
    else:
        atoms = subset_structure_to_atom_type(atoms, granularity)

    protein_df = atoms

    # filter hetero-atoms
    if keep_hets:
        hetatms_to_keep = filter_hetatms(atoms, keep_hets)
        protein_df = pd.concat([atoms, hetatms_to_keep])

    # remove alternative-location (`alt_loc`) residues
    if not insertions:
        protein_df = remove_insertions(protein_df)

    # perform chain selection
    protein_df = select_chains(protein_df, chain_selection=chain_selection)
    return protein_df


@typechecked
def compute_covalent_bond_matrix(pdb_filepath: str, tolerance: float = TOLERANCE) -> np.ndarray:
    raw_pdb = read_pdb_file_to_pdb(
        pdb_path=pdb_filepath,
        verbose=False,
    )

    processed_pdb_df = process_pdb(
        protein_pdb=raw_pdb,
        atom_df_processing_funcs=None,
        hetatom_df_processing_funcs=None,
        granularity="atom",
        chain_selection="all",
        insertions=False,
        deprotonate=True,
        keep_hets=[]
    )

    dist_mat = compute_distmat(processed_pdb_df)

    # assign bond states to the dataframe, and then map these to covalent radii
    processed_pdb_df = assign_bond_states_to_dataframe(processed_pdb_df)
    processed_pdb_df = assign_covalent_radii_to_dataframe(processed_pdb_df)

    # create a covalent distance matrix by adding the radius array with its transpose
    covalent_radius_distance_matrix = np.add(
        np.array(processed_pdb_df["covalent_radius"]).reshape(-1, 1),
        np.array(processed_pdb_df["covalent_radius"]).reshape(1, -1),
    )

    # add the tolerance
    covalent_radius_distance_matrix = (covalent_radius_distance_matrix + tolerance)

    # threshold distance matrix to entries where the distance is less than the covalent radius plus tolerance and also larger than `0.4`
    dist_mat = dist_mat[dist_mat > 0.4]
    t_bond_mat = dist_mat[dist_mat < covalent_radius_distance_matrix]
    t_bond_mat = np.nan_to_num(t_bond_mat)
    t_bond_mat[t_bond_mat > 0] = 1

    return t_bond_mat
