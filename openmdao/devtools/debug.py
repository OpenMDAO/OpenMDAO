"""Various debugging aids."""

import sys

def allproc_var_dump(system, out=sys.stdout):
    """Dump various variable info across all procs.

    Args
    ----
    system : System
        System to dump variable info for.

    out : stream (sys.stdout)
        Output stream to write to.

    """
    assembler = system._sys_assembler
    set_IDs = assembler._variable_set_IDs
    set_idxs = assembler._variable_set_indices

    for typ in ['input', 'output']:
        vset_indices = assembler._variable_set_indices[typ]
        vset_IDs = assembler._variable_set_IDs[typ]
        allproc_indices = system._variable_allprocs_indices[typ]

        out.write("\n%s %ss:\n" % (system.path_name, typ))
        lens = [len(n) for n in system._variable_allprocs_names[typ]]
        nwid = max(lens) if lens else 0
        nwid = max(nwid, len("Name"))
        args = (
            "Name",
            "g_idx",
            "vset",
            "vset_idx"
        )
        wids = [len(a) for a in args]
        wids[0] = nwid

        template = ' '.join(["{%d:<%d}" % (i,wid) for i,wid in enumerate(wids)])
        template += "\n"
        out.write(template.format(*args))
        for idx, name in enumerate(system._variable_allprocs_names[typ]):
            out.write(template.format(name,
                                allproc_indices[name],
                                vset_IDs[vset_indices[idx][0]],
                                vset_indices[idx][1],
                                nwid=nwid))
