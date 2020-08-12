
    self._setup_var_data()

    def _setup_var_data(self):
        """
        Compute the list of abs var names, abs/prom name maps, and metadata dictionaries.
        """
        self._var_allprocs_abs_names = {'input': [], 'output': []}
        self._var_abs_names = {'input': [], 'output': []}
        self._var_allprocs_prom2abs_list = {'input': OrderedDict(), 'output': OrderedDict()}
        self._var_abs2prom = {'input': {}, 'output': {}}
        self._var_allprocs_abs2prom = {'input': {}, 'output': {}}
        self._var_allprocs_abs2meta = {}
        self._var_abs2meta = {}
        self._var_allprocs_abs2idx = {}



    # creates and populates _var_* dictionaries

    def _setup_var_data(self):
        """
        Compute the list of abs var names, abs/prom name maps, and metadata dictionaries.
        """
        super(Group, self)._setup_var_data()


        abs_names = self._var_abs_names
        abs_names_discrete = self._var_abs_names_discrete

        allprocs_abs_names = self._var_allprocs_abs_names
        allprocs_abs_names_discrete = self._var_allprocs_abs_names_discrete

        var_discrete = self._var_discrete
        allprocs_discrete = self._var_allprocs_discrete

        abs2meta = self._var_abs2meta
        abs2prom = self._var_abs2prom

        allprocs_abs2meta = self._var_allprocs_abs2meta
        allprocs_abs2prom = self._var_allprocs_abs2prom

        allprocs_prom2abs_list = self._var_allprocs_prom2abs_list

        group_inputs = []
        for n, meta in self._group_inputs.items():
            meta['path'] = self.pathname  # used for error reporting

        for subsys in self._subsystems_myproc:
            subsys._setup_var_data()
            self._has_output_scaling |= subsys._has_output_scaling
            self._has_resid_scaling |= subsys._has_resid_scaling

            var_maps = subsys._get_maps(subsys._var_allprocs_prom2abs_list)

            # Assemble allprocs_abs2meta and abs2meta
            allprocs_abs2meta.update(subsys._var_allprocs_abs2meta)
            abs2meta.update(subsys._var_abs2meta)

            sub_prefix = subsys.name + '.'

            for type_ in ['input', 'output']:
                # Assemble abs_names and allprocs_abs_names
                allprocs_abs_names[type_].extend(
                    subsys._var_allprocs_abs_names[type_])
                allprocs_abs_names_discrete[type_].extend(
                    subsys._var_allprocs_abs_names_discrete[type_])

                abs_names[type_].extend(subsys._var_abs_names[type_])
                abs_names_discrete[type_].extend(subsys._var_abs_names_discrete[type_])

                allprocs_discrete[type_].update({k: v for k, v in
                                                 subsys._var_allprocs_discrete[type_].items()})
                var_discrete[type_].update({sub_prefix + k: v for k, v in
                                            subsys._var_discrete[type_].items()})

                # Assemble abs2prom
                sub_loc_proms = subsys._var_abs2prom[type_]
                sub_proms = subsys._var_allprocs_abs2prom[type_]
                for abs_name in chain(subsys._var_allprocs_abs_names[type_],
                                      subsys._var_allprocs_abs_names_discrete[type_]):
                    if abs_name in sub_loc_proms:
                        abs2prom[type_][abs_name] = var_maps[type_][sub_loc_proms[abs_name]]

                    allprocs_abs2prom[type_][abs_name] = var_maps[type_][sub_proms[abs_name]]

                # Assemble allprocs_prom2abs_list
                for sub_prom, sub_abs in subsys._var_allprocs_prom2abs_list[type_].items():
                    prom_name = var_maps[type_][sub_prom]
                    if prom_name not in allprocs_prom2abs_list[type_]:
                        allprocs_prom2abs_list[type_][prom_name] = []
                    allprocs_prom2abs_list[type_][prom_name].extend(sub_abs)
                    if type_ == 'input' and isinstance(subsys, Group):
                        if sub_prom in subsys._group_inputs:
                            group_inputs.append((prom_name, subsys._group_inputs[sub_prom]))

        for prom_name, abs_list in allprocs_prom2abs_list['output'].items():
            if len(abs_list) > 1:
                raise RuntimeError("{}: Output name '{}' refers to "
                                   "multiple outputs: {}.".format(self.msginfo, prom_name,
                                                                  sorted(abs_list)))

        # If running in parallel, allgather
        if self.comm.size > 1:
            mysub = self._subsystems_myproc[0] if self._subsystems_myproc else False
            if (mysub and mysub.comm.rank == 0 and (mysub._full_comm is None or
                                                    mysub._full_comm.rank == 0)):
                raw = (allprocs_abs_names, allprocs_discrete, allprocs_prom2abs_list,
                       allprocs_abs2prom, allprocs_abs2meta, self._has_output_scaling,
                       self._has_resid_scaling, group_inputs)
            else:
                raw = (
                    {'input': [], 'output': []},
                    {'input': {}, 'output': {}},
                    {'input': {}, 'output': {}},
                    {'input': {}, 'output': {}},
                    {},
                    False,
                    False,
                    []
                )
            gathered = self.comm.allgather(raw)

            for type_ in ['input', 'output']:
                allprocs_abs_names[type_] = []
                allprocs_abs2prom[type_] = {}
                allprocs_prom2abs_list[type_] = OrderedDict()

            group_inputs = []
            for (myproc_abs_names, myproc_discrete, myproc_prom2abs_list, all_abs2prom,
                 myproc_abs2meta, oscale, rscale, ginputs) in gathered:
                self._has_output_scaling |= oscale
                self._has_resid_scaling |= rscale

                group_inputs.extend(ginputs)

                # Assemble in parallel allprocs_abs2meta
                for n in myproc_abs2meta:
                    if n not in allprocs_abs2meta:
                        allprocs_abs2meta[n] = myproc_abs2meta[n]

                for type_ in ['input', 'output']:

                    # Assemble in parallel allprocs_abs_names
                    allprocs_abs_names[type_].extend(myproc_abs_names[type_])
                    allprocs_discrete[type_].update(myproc_discrete[type_])
                    allprocs_abs2prom[type_].update(all_abs2prom[type_])

                    # Assemble in parallel allprocs_prom2abs_list
                    for prom_name, abs_names_list in myproc_prom2abs_list[type_].items():
                        if prom_name not in allprocs_prom2abs_list[type_]:
                            allprocs_prom2abs_list[type_][prom_name] = []
                        allprocs_prom2abs_list[type_][prom_name].extend(abs_names_list)

        ginputs = self._group_inputs
        for prom, meta in group_inputs:
            if prom in ginputs:
                # check for any conflicting units or values
                old = ginputs[prom]

                for n, val in meta.items():
                    if n == 'path' or val is None:
                        continue

                    if n in old and old[n] is not None:
                        if isinstance(val, np.ndarray) or isinstance(old[n], np.ndarray):
                            eq = np.all(val == old[n])
                        else:
                            eq = val == old[n]

                        if not eq:
                            raise RuntimeError(f"Groups '{old['path']}' and '{meta['path']}' "
                                               f"added the input '{prom}' with conflicting '{n}'.")
                    old[n] = val
            else:
                ginputs[prom] = meta

        if ginputs:
            extra = set(ginputs).difference(self._var_allprocs_prom2abs_list['input'])
            if extra:
                raise RuntimeError(f"{self.msginfo}: The following group inputs could not be "
                                   f"found: {sorted(extra)}.")

        if self._var_discrete['input'] or self._var_discrete['output']:
            self._discrete_inputs = _DictValues(self._var_discrete['input'])
            self._discrete_outputs = _DictValues(self._var_discrete['output'])
        else:
            self._discrete_inputs = self._discrete_outputs = ()


    def _setup_var_data(self):
        """
        Compute the list of abs var names, abs/prom name maps, and metadata dictionaries.
        """
        global global_meta_names
        super(Component, self)._setup_var_data()

        allprocs_abs_names = self._var_allprocs_abs_names
        allprocs_abs_names_discrete = self._var_allprocs_abs_names_discrete

        allprocs_prom2abs_list = self._var_allprocs_prom2abs_list

        abs2prom = self._var_abs2prom

        allprocs_abs2meta = self._var_allprocs_abs2meta
        abs2meta = self._var_abs2meta

        # Compute the prefix for turning rel/prom names into abs names
        prefix = self.pathname + '.' if self.pathname else ''

        for type_ in ['input', 'output']:
            for prom_name in self._var_rel_names[type_]:
                abs_name = prefix + prom_name
                metadata = self._var_rel2meta[prom_name]

                # Compute allprocs_abs_names
                allprocs_abs_names[type_].append(abs_name)

                # Compute allprocs_prom2abs_list, abs2prom
                allprocs_prom2abs_list[type_][prom_name] = [abs_name]
                abs2prom[type_][abs_name] = prom_name

                # Compute allprocs_abs2meta
                allprocs_abs2meta[abs_name] = {
                    meta_name: metadata[meta_name]
                    for meta_name in global_meta_names[type_]
                }

                # Compute abs2meta
                abs2meta[abs_name] = metadata

            for prom_name, val in self._var_discrete[type_].items():
                abs_name = prefix + prom_name

                # Compute allprocs_abs_names_discrete
                allprocs_abs_names_discrete[type_].append(abs_name)

                # Compute allprocs_prom2abs_list, abs2prom
                allprocs_prom2abs_list[type_][prom_name] = [abs_name]
                abs2prom[type_][abs_name] = prom_name

                # Compute allprocs_discrete (metadata for discrete vars)
                self._var_allprocs_discrete[type_][abs_name] = v = val.copy()
                del v['value']

        self._var_allprocs_abs2prom = abs2prom

        self._var_abs_names = allprocs_abs_names
        self._var_abs_names_discrete = allprocs_abs_names_discrete

        if self._var_discrete['input'] or self._var_discrete['output']:
            self._discrete_inputs = _DictValues(self._var_discrete['input'])
            self._discrete_outputs = _DictValues(self._var_discrete['output'])
        else:
            self._discrete_inputs = self._discrete_outputs = ()


    self._setup_vec_names(mode, self._vec_names, self._vois)

    

    # sets  self._vec_names =  ['nonlinear', 'linear'] + any response vars

    def _setup_vec_names(self, mode, vec_names=None, vois=None):
        """
        Return the list of vec_names and the vois dict.

        Parameters
        ----------
        mode : str
            Derivative direction, either 'fwd' or 'rev'.
        vec_names : list of str or None
            The list of names of vectors. Depends on the value of mode.
        vois : dict
            Dictionary of either design vars or responses, depending on the value
            of mode.

        """
        self._vois = vois
        if vec_names is None:  # should only occur at top level on full setup
            if self._use_derivatives:
                vec_names = ['nonlinear', 'linear']
                if mode == 'fwd':
                    self._vois = vois = self.get_design_vars(recurse=True, get_sizes=False)
                else:  # rev
                    self._vois = vois = self.get_responses(recurse=True, get_sizes=False)
                vec_names.extend(sorted(set(voi for voi, data in vois.items()
                                            if data['parallel_deriv_color'] is not None
                                            or data['vectorize_derivs'])))
            else:
                vec_names = ['nonlinear']
                self._vois = {}

        self._vec_names = vec_names
        self._lin_vec_names = vec_names[1:]  # only linear vec names

        for s in self.system_iter():
            s._vec_names = vec_names
            s._lin_vec_names = self._lin_vec_names


        def _setup_global_connections(self, conns=None):
        """
        Compute dict of all connections between this system's inputs and outputs.

        The connections come from 4 sources:
        1. Implicit connections owned by the current system
        2. Explicit connections declared by the current system
        3. Explicit connections declared by parent systems
        4. Implicit / explicit from subsystems

        Parameters
        ----------
        conns : dict
            Dictionary of connections passed down from parent group.
        """
        if self._raise_connection_errors is False:
            self._set_subsys_connection_errors(False)

        global_abs_in2out = self._conn_global_abs_in2out = {}

        allprocs_prom2abs_list_in = self._var_allprocs_prom2abs_list['input']
        allprocs_prom2abs_list_out = self._var_allprocs_prom2abs_list['output']

        allprocs_discrete_in = self._var_allprocs_discrete['input']
        allprocs_discrete_out = self._var_allprocs_discrete['output']

        abs2meta = self._var_abs2meta
        pathname = self.pathname

        abs_in2out = {}

        if pathname == '':
            path_len = 0
            nparts = 0
        else:
            path_len = len(pathname) + 1
            nparts = len(pathname.split('.'))

        new_conns = defaultdict(dict)

        if conns is not None:
            for abs_in, abs_out in conns.items():
                inparts = abs_in.split('.')
                outparts = abs_out.split('.')

                if inparts[:nparts] == outparts[:nparts]:
                    global_abs_in2out[abs_in] = abs_out

                    # if connection is contained in a subgroup, add to conns
                    # to pass down to subsystems.
                    if inparts[:nparts + 1] == outparts[:nparts + 1]:
                        new_conns[inparts[nparts]][abs_in] = abs_out

        # Add implicit connections (only ones owned by this group)
        for prom_name in allprocs_prom2abs_list_out:
            if prom_name in allprocs_prom2abs_list_in:
                abs_out = allprocs_prom2abs_list_out[prom_name][0]
                out_subsys = abs_out[path_len:].split('.', 1)[0]
                for abs_in in allprocs_prom2abs_list_in[prom_name]:
                    in_subsys = abs_in[path_len:].split('.', 1)[0]
                    if out_subsys != in_subsys:
                        abs_in2out[abs_in] = abs_out

        # Add explicit connections (only ones declared by this group)
        for prom_in, (prom_out, src_indices, flat_src_indices) in \
                self._manual_connections.items():

            # throw an exception if either output or input doesn't exist
            # (not traceable to a connect statement, so provide context)
            if not (prom_out in allprocs_prom2abs_list_out or prom_out in allprocs_discrete_out):
                if (prom_out in allprocs_prom2abs_list_in or prom_out in allprocs_discrete_in):
                    msg = f"{self.msginfo}: Attempted to connect from '{prom_out}' to " + \
                          f"'{prom_in}', but '{prom_out}' is an input. " + \
                          "All connections must be from an output to an input."
                    if self._raise_connection_errors:
                        raise NameError(msg)
                    else:
                        simple_warning(msg)
                        continue
                else:
                    msg = f"{self.msginfo}: Attempted to connect from '{prom_out}' to " + \
                          f"'{prom_in}', but '{prom_out}' doesn't exist."
                    if self._raise_connection_errors:
                        raise NameError(msg)
                    else:
                        simple_warning(msg)
                        continue

            if not (prom_in in allprocs_prom2abs_list_in or prom_in in allprocs_discrete_in):
                if (prom_in in allprocs_prom2abs_list_out or prom_in in allprocs_discrete_out):
                    msg = f"{self.msginfo}: Attempted to connect from '{prom_out}' to " + \
                          f"'{prom_in}', but '{prom_in}' is an output. " + \
                          "All connections must be from an output to an input."
                    if self._raise_connection_errors:
                        raise NameError(msg)
                    else:
                        simple_warning(msg)
                        continue
                else:
                    msg = f"{self.msginfo}: Attempted to connect from '{prom_out}' to " + \
                          f"'{prom_in}', but '{prom_in}' doesn't exist."
                    if self._raise_connection_errors:
                        raise NameError(msg)
                    else:
                        simple_warning(msg)
                        continue

            # Throw an exception if output and input are in the same system
            # (not traceable to a connect statement, so provide context)
            # and check if src_indices is defined in both connect and add_input.
            abs_out = allprocs_prom2abs_list_out[prom_out][0]
            outparts = abs_out.split('.')
            out_subsys = outparts[:-1]

            for abs_in in allprocs_prom2abs_list_in[prom_in]:
                inparts = abs_in.split('.')
                in_subsys = inparts[:-1]
                if out_subsys == in_subsys:
                    msg = f"{self.msginfo}: Output and input are in the same System for " + \
                          f"connection from '{prom_out}' to '{prom_in}'."
                    if self._raise_connection_errors:
                        raise RuntimeError(msg)
                    else:
                        simple_warning(msg)
                        continue

                if src_indices is not None and abs_in in abs2meta:
                    meta = abs2meta[abs_in]
                    if meta['src_indices'] is not None:
                        msg = f"{self.msginfo}: src_indices has been defined in both " + \
                              f"connect('{prom_out}', '{prom_in}') and " + \
                              f"add_input('{prom_in}', ...)."
                        if self._raise_connection_errors:
                            raise RuntimeError(msg)
                        else:
                            simple_warning(msg)
                            continue
                    meta['src_indices'] = np.atleast_1d(src_indices)
                    meta['flat_src_indices'] = flat_src_indices

                if abs_in in abs_in2out:
                    msg = f"{self.msginfo}: Input '{abs_in}' cannot be connected to " + \
                          f"'{abs_out}' because it's already connected to '{abs_in2out[abs_in]}'"
                    if self._raise_connection_errors:
                        raise RuntimeError(msg)
                    else:
                        simple_warning(msg)
                        continue

                abs_in2out[abs_in] = abs_out

                # if connection is contained in a subgroup, add to conns to pass down to subsystems.
                if inparts[:nparts + 1] == outparts[:nparts + 1]:
                    new_conns[inparts[nparts]][abs_in] = abs_out

        # Recursion
        distcomps = []
        for subsys in self._subsystems_myproc:
            if isinstance(subsys, Group):
                if subsys.name in new_conns:
                    subsys._setup_global_connections(conns=new_conns[subsys.name])
                else:
                    subsys._setup_global_connections()
            elif subsys.options['distributed'] and subsys.comm.size > 1:
                distcomps.append(subsys)

        # Compute global_abs_in2out by first adding this group's contributions,
        # then adding contributions from systems above/below, then allgathering.
        conn_list = list(global_abs_in2out.items())
        conn_list.extend(abs_in2out.items())
        global_abs_in2out.update(abs_in2out)

        for subsys in self._subgroups_myproc:
            global_abs_in2out.update(subsys._conn_global_abs_in2out)
            conn_list.extend(subsys._conn_global_abs_in2out.items())

        if len(conn_list) > len(global_abs_in2out):
            dupes = [n for n, val in Counter(tgt for tgt, src in conn_list).items() if val > 1]
            dup_info = defaultdict(set)
            for tgt, src in conn_list:
                for dup in dupes:
                    if tgt == dup:
                        dup_info[tgt].add(src)
            dup_info = [(n, srcs) for n, srcs in dup_info.items() if len(srcs) > 1]
            if dup_info:
                dup = ["%s from %s" % (tgt, sorted(srcs)) for tgt, srcs in dup_info]
                msg = f"{self.msginfo}: The following inputs have multiple connections: " + \
                      f"{', '.join(dup)}"
                if self._raise_connection_errors:
                    raise RuntimeError(msg)
                else:
                    simple_warning(msg)

        # If running in parallel, allgather
        if self.comm.size > 1:
            if self._subsystems_myproc and self._subsystems_myproc[0].comm.rank == 0:
                raw = global_abs_in2out
            else:
                raw = {}
            gathered = self.comm.allgather(raw)

            for myproc_global_abs_in2out in gathered:
                global_abs_in2out.update(myproc_global_abs_in2out)

            for comp in distcomps:
                comp._update_dist_src_indices(global_abs_in2out)

    # adds connections to self._conn_global_abs_in2out

    def _setup_global_connections(self, conns=None):
        """
        Compute dict of all connections between this system's inputs and outputs.

        The connections come from 4 sources:
        1. Implicit connections owned by the current system
        2. Explicit connections declared by the current system
        3. Explicit connections declared by parent systems
        4. Implicit / explicit from subsystems

        Parameters
        ----------
        conns : dict
            Dictionary of connections passed down from parent group.
        """
        if self._raise_connection_errors is False:
            self._set_subsys_connection_errors(False)

        global_abs_in2out = self._conn_global_abs_in2out = {}

        allprocs_prom2abs_list_in = self._var_allprocs_prom2abs_list['input']
        allprocs_prom2abs_list_out = self._var_allprocs_prom2abs_list['output']

        allprocs_discrete_in = self._var_allprocs_discrete['input']
        allprocs_discrete_out = self._var_allprocs_discrete['output']

        abs2meta = self._var_abs2meta
        pathname = self.pathname

        abs_in2out = {}

        if pathname == '':
            path_len = 0
            nparts = 0
        else:
            path_len = len(pathname) + 1
            nparts = len(pathname.split('.'))

        new_conns = defaultdict(dict)

        if conns is not None:
            for abs_in, abs_out in conns.items():
                inparts = abs_in.split('.')
                outparts = abs_out.split('.')

                if inparts[:nparts] == outparts[:nparts]:
                    global_abs_in2out[abs_in] = abs_out

                    # if connection is contained in a subgroup, add to conns
                    # to pass down to subsystems.
                    if inparts[:nparts + 1] == outparts[:nparts + 1]:
                        new_conns[inparts[nparts]][abs_in] = abs_out

        # Add implicit connections (only ones owned by this group)
        for prom_name in allprocs_prom2abs_list_out:
            if prom_name in allprocs_prom2abs_list_in:
                abs_out = allprocs_prom2abs_list_out[prom_name][0]
                out_subsys = abs_out[path_len:].split('.', 1)[0]
                for abs_in in allprocs_prom2abs_list_in[prom_name]:
                    in_subsys = abs_in[path_len:].split('.', 1)[0]
                    if out_subsys != in_subsys:
                        abs_in2out[abs_in] = abs_out

        # Add explicit connections (only ones declared by this group)
        for prom_in, (prom_out, src_indices, flat_src_indices) in \
                self._manual_connections.items():

            # throw an exception if either output or input doesn't exist
            # (not traceable to a connect statement, so provide context)
            if not (prom_out in allprocs_prom2abs_list_out or prom_out in allprocs_discrete_out):
                if (prom_out in allprocs_prom2abs_list_in or prom_out in allprocs_discrete_in):
                    msg = f"{self.msginfo}: Attempted to connect from '{prom_out}' to " + \
                          f"'{prom_in}', but '{prom_out}' is an input. " + \
                          "All connections must be from an output to an input."
                    if self._raise_connection_errors:
                        raise NameError(msg)
                    else:
                        simple_warning(msg)
                        continue
                else:
                    msg = f"{self.msginfo}: Attempted to connect from '{prom_out}' to " + \
                          f"'{prom_in}', but '{prom_out}' doesn't exist."
                    if self._raise_connection_errors:
                        raise NameError(msg)
                    else:
                        simple_warning(msg)
                        continue

            if not (prom_in in allprocs_prom2abs_list_in or prom_in in allprocs_discrete_in):
                if (prom_in in allprocs_prom2abs_list_out or prom_in in allprocs_discrete_out):
                    msg = f"{self.msginfo}: Attempted to connect from '{prom_out}' to " + \
                          f"'{prom_in}', but '{prom_in}' is an output. " + \
                          "All connections must be from an output to an input."
                    if self._raise_connection_errors:
                        raise NameError(msg)
                    else:
                        simple_warning(msg)
                        continue
                else:
                    msg = f"{self.msginfo}: Attempted to connect from '{prom_out}' to " + \
                          f"'{prom_in}', but '{prom_in}' doesn't exist."
                    if self._raise_connection_errors:
                        raise NameError(msg)
                    else:
                        simple_warning(msg)
                        continue

            # Throw an exception if output and input are in the same system
            # (not traceable to a connect statement, so provide context)
            # and check if src_indices is defined in both connect and add_input.
            abs_out = allprocs_prom2abs_list_out[prom_out][0]
            outparts = abs_out.split('.')
            out_subsys = outparts[:-1]

            for abs_in in allprocs_prom2abs_list_in[prom_in]:
                inparts = abs_in.split('.')
                in_subsys = inparts[:-1]
                if out_subsys == in_subsys:
                    msg = f"{self.msginfo}: Output and input are in the same System for " + \
                          f"connection from '{prom_out}' to '{prom_in}'."
                    if self._raise_connection_errors:
                        raise RuntimeError(msg)
                    else:
                        simple_warning(msg)
                        continue

                if src_indices is not None and abs_in in abs2meta:
                    meta = abs2meta[abs_in]
                    if meta['src_indices'] is not None:
                        msg = f"{self.msginfo}: src_indices has been defined in both " + \
                              f"connect('{prom_out}', '{prom_in}') and " + \
                              f"add_input('{prom_in}', ...)."
                        if self._raise_connection_errors:
                            raise RuntimeError(msg)
                        else:
                            simple_warning(msg)
                            continue
                    meta['src_indices'] = np.atleast_1d(src_indices)
                    meta['flat_src_indices'] = flat_src_indices

                if abs_in in abs_in2out:
                    msg = f"{self.msginfo}: Input '{abs_in}' cannot be connected to " + \
                          f"'{abs_out}' because it's already connected to '{abs_in2out[abs_in]}'"
                    if self._raise_connection_errors:
                        raise RuntimeError(msg)
                    else:
                        simple_warning(msg)
                        continue

                abs_in2out[abs_in] = abs_out

                # if connection is contained in a subgroup, add to conns to pass down to subsystems.
                if inparts[:nparts + 1] == outparts[:nparts + 1]:
                    new_conns[inparts[nparts]][abs_in] = abs_out

        # Recursion
        distcomps = []
        for subsys in self._subsystems_myproc:
            if isinstance(subsys, Group):
                if subsys.name in new_conns:
                    subsys._setup_global_connections(conns=new_conns[subsys.name])
                else:
                    subsys._setup_global_connections()
            elif subsys.options['distributed'] and subsys.comm.size > 1:
                distcomps.append(subsys)

        # Compute global_abs_in2out by first adding this group's contributions,
        # then adding contributions from systems above/below, then allgathering.
        conn_list = list(global_abs_in2out.items())
        conn_list.extend(abs_in2out.items())
        global_abs_in2out.update(abs_in2out)

        for subsys in self._subgroups_myproc:
            global_abs_in2out.update(subsys._conn_global_abs_in2out)
            conn_list.extend(subsys._conn_global_abs_in2out.items())

        if len(conn_list) > len(global_abs_in2out):
            dupes = [n for n, val in Counter(tgt for tgt, src in conn_list).items() if val > 1]
            dup_info = defaultdict(set)
            for tgt, src in conn_list:
                for dup in dupes:
                    if tgt == dup:
                        dup_info[tgt].add(src)
            dup_info = [(n, srcs) for n, srcs in dup_info.items() if len(srcs) > 1]
            if dup_info:
                dup = ["%s from %s" % (tgt, sorted(srcs)) for tgt, srcs in dup_info]
                msg = f"{self.msginfo}: The following inputs have multiple connections: " + \
                      f"{', '.join(dup)}"
                if self._raise_connection_errors:
                    raise RuntimeError(msg)
                else:
                    simple_warning(msg)

        # If running in parallel, allgather
        if self.comm.size > 1:
            if self._subsystems_myproc and self._subsystems_myproc[0].comm.rank == 0:
                raw = global_abs_in2out
            else:
                raw = {}
            gathered = self.comm.allgather(raw)

            for myproc_global_abs_in2out in gathered:
                global_abs_in2out.update(myproc_global_abs_in2out)

            for comp in distcomps:
                comp._update_dist_src_indices(global_abs_in2out)


    # sets up self._var_allprocs_relevant_name

    self._setup_relevance(mode, self._relevant)

    def _setup_relevance(self, mode, relevant=None):
        """
        Set up the relevance dictionary.

        Parameters
        ----------
        mode : str
            Derivative direction, either 'fwd' or 'rev'.
        relevant : dict or None
            Dictionary mapping VOI name to all variables necessary for computing
            derivatives between the VOI and all other VOIs.
        """
        if relevant is None:  # should only occur at top level on full setup
            self._relevant = relevant = self._init_relevance(mode)
        else:
            self._relevant = relevant

        self._var_allprocs_relevant_names = defaultdict(lambda: {'input': [], 'output': []})
        self._var_relevant_names = defaultdict(lambda: {'input': [], 'output': []})

        self._rel_vec_name_list = []
        for vec_name in self._vec_names:
            rel, relsys = relevant[vec_name]['@all']
            if self.pathname in relsys:
                self._rel_vec_name_list.append(vec_name)
            for type_ in ('input', 'output'):
                self._var_allprocs_relevant_names[vec_name][type_].extend(
                    v for v in self._var_allprocs_abs_names[type_] if v in rel[type_])
                self._var_relevant_names[vec_name][type_].extend(
                    v for v in self._var_abs_names[type_] if v in rel[type_])

        self._rel_vec_names = frozenset(self._rel_vec_name_list)
        self._lin_rel_vec_name_list = self._rel_vec_name_list[1:]

        for s in self._subsystems_myproc:
            s._setup_relevance(mode, relevant)


    # sets up self._subsystems_var_range

    def _setup_var_index_ranges(self):
        """
        Compute the division of variables by subsystem.
        """
        nsub_allprocs = len(self._subsystems_allprocs)

        subsystems_var_range = self._subsystems_var_range = {}

        vec_names = self._lin_rel_vec_name_list if self._use_derivatives else self._vec_names

        # First compute these on one processor for each subsystem
        for vec_name in vec_names:

            # Here, we count the number of variables in each subsystem.
            # We do this so that we can compute the offset when we recurse into each subsystem.
            allprocs_counters = {}
            for type_ in ['input', 'output']:
                allprocs_counters[type_] = np.zeros(nsub_allprocs, INT_DTYPE)
                for subsys in self._subsystems_myproc:
                    comm = subsys.comm if subsys._full_comm is None else subsys._full_comm
                    if comm.rank == 0 and vec_name in subsys._rel_vec_names:
                        isub = self._subsystems_inds[subsys.name]
                        allprocs_counters[type_][isub] = \
                            len(subsys._var_allprocs_relevant_names[vec_name][type_])

            # If running in parallel, allgather
            if self.comm.size > 1:
                gathered = self.comm.allgather(allprocs_counters)
                allprocs_counters = {
                    type_: np.zeros(nsub_allprocs, INT_DTYPE) for type_ in ['input', 'output']
                }
                for myproc_counters in gathered:
                    for type_ in ['input', 'output']:
                        allprocs_counters[type_] += myproc_counters[type_]

            # Compute _subsystems_var_range
            subsystems_var_range[vec_name] = {}

            for type_ in ['input', 'output']:
                subsystems_var_range[vec_name][type_] = {}

                for subsys in self._subsystems_myproc:
                    if vec_name not in subsys._rel_vec_names:
                        continue
                    isub = self._subsystems_inds[subsys.name]
                    start = np.sum(allprocs_counters[type_][:isub])
                    subsystems_var_range[vec_name][type_][subsys.name] = (
                        start, start + allprocs_counters[type_][isub]
                    )

        if self._use_derivatives:
            subsystems_var_range['nonlinear'] = subsystems_var_range['linear']

        self._setup_var_index_maps()

        for subsys in self._subsystems_myproc:
            subsys._setup_var_index_ranges()



    def _setup_var_index_maps(self):
        """
        Compute maps from abs var names to their index among allprocs variables in this system.
        """
        self._var_allprocs_abs2idx = abs2idx = {}

        vec_names = self._lin_rel_vec_name_list if self._use_derivatives else self._vec_names

        for vec_name in vec_names:
            abs2idx[vec_name] = abs2idx_t = {}
            for type_ in ['input', 'output']:
                for i, abs_name in enumerate(self._var_allprocs_relevant_names[vec_name][type_]):
                    abs2idx_t[abs_name] = i

        if self._use_derivatives:
            abs2idx['nonlinear'] = abs2idx['linear']

        for subsys in self._subsystems_myproc:
            subsys._setup_var_index_maps()



    def _setup_var_sizes(self):
        """
        Compute the arrays of local variable sizes for all variables/procs on this system.
        """
        super(Group, self)._setup_var_sizes()

        self._var_offsets = None

        iproc = self.comm.rank
        nproc = self.comm.size

        subsystems_proc_range = self._subsystems_proc_range

        # Recursion
        for subsys in self._subsystems_myproc:
            subsys._setup_var_sizes()

        sizes = self._var_sizes
        relnames = self._var_allprocs_relevant_names

        vec_names = self._lin_rel_vec_name_list if self._use_derivatives else self._vec_names

        n_distrib_vars = 0
        n_parallel_sub = 0

        # Compute _var_sizes
        for vec_name in vec_names:
            sizes[vec_name] = {}
            subsystems_var_range = self._subsystems_var_range[vec_name]

            for type_ in ['input', 'output']:
                sizes[vec_name][type_] = sz = np.zeros((nproc, len(relnames[vec_name][type_])),
                                                       INT_DTYPE)

                for ind, subsys in enumerate(self._subsystems_myproc):
                    if isinstance(subsys, Component):
                        if subsys.options['distributed']:
                            n_distrib_vars += 1
                    elif subsys._has_distrib_vars:
                        n_distrib_vars += 1
                    elif subsys._contains_parallel_group or subsys._mpi_proc_allocator.parallel:
                        n_parallel_sub += 1

                    if vec_name not in subsys._rel_vec_names:
                        continue
                    proc_slice = slice(*subsystems_proc_range[ind])
                    var_slice = slice(*subsystems_var_range[type_][subsys.name])
                    if proc_slice.stop - proc_slice.start > subsys.comm.size:
                        # in this case, we've split the proc for parallel FD, so subsys doesn't
                        # have var_sizes for all the ranks we need. Since each parallel FD comm
                        # has the same size distribution (since all are identical), just 'tile'
                        # the var_sizes from the subsystem to fill in the full rank range we need
                        # at this level.
                        assert (proc_slice.stop - proc_slice.start) % subsys.comm.size == 0, \
                            "%s comm size (%d) is not an exact multiple of %s comm size (%d)" % (
                                self.pathname, self.comm.size, subsys.pathname, subsys.comm.size)
                        proc_i = proc_slice.start
                        while proc_i < proc_slice.stop:
                            sz[proc_i:proc_i + subsys.comm.size, var_slice] = \
                                subsys._var_sizes[vec_name][type_]
                            proc_i += subsys.comm.size
                    else:
                        sz[proc_slice, var_slice] = subsys._var_sizes[vec_name][type_]

        # If parallel, all gather
        if self.comm.size > 1:
            for vec_name in self._lin_rel_vec_name_list:
                sizes = self._var_sizes[vec_name]
                for type_ in ['input', 'output']:
                    sizes_in = sizes[type_][iproc, :].copy()
                    self.comm.Allgather(sizes_in, sizes[type_])

            self._has_distrib_vars = self.comm.allreduce(n_distrib_vars) > 0
            self._contains_parallel_group = self.comm.allreduce(n_parallel_sub) > 0

            if (self._has_distrib_vars or self._contains_parallel_group or
                not np.all(self._var_sizes[vec_names[0]]['output']) or
               not np.all(self._var_sizes[vec_names[0]]['input'])):

                if self._distributed_vector_class is not None:
                    self._vector_class = self._distributed_vector_class
                else:
                    raise RuntimeError("{}: Distributed vectors are required but no distributed "
                                       "vector type has been set.".format(self.msginfo))

            # compute owning ranks and owned sizes
            abs2meta = self._var_allprocs_abs2meta
            owns = self._owning_rank
            self._owned_sizes = self._var_sizes[vec_names[0]]['output'].copy()
            for type_ in ('input', 'output'):
                sizes = self._var_sizes[vec_names[0]][type_]
                for i, name in enumerate(self._var_allprocs_abs_names[type_]):
                    for rank in range(self.comm.size):
                        if sizes[rank, i] > 0:
                            owns[name] = rank
                            if type_ == 'output' and not abs2meta[name]['distributed']:
                                self._owned_sizes[rank + 1:, i] = 0  # zero out all dups
                            break

                if self._var_allprocs_discrete[type_]:
                    local = list(self._var_discrete[type_])
                    for i, names in enumerate(self.comm.allgather(local)):
                        for n in names:
                            if n not in owns:
                                owns[n] = i
        else:
            self._owned_sizes = self._var_sizes[vec_names[0]]['output']
            self._vector_class = self._local_vector_class

        if self._use_derivatives:
            self._var_sizes['nonlinear'] = self._var_sizes['linear']

        if self.comm.size > 1:
            self._setup_global_shapes()


        if self.pathname == '':
            self._resolve_connected_input_defaults()


    def _resolve_connected_input_defaults(self):
        conns = self._conn_global_abs_in2out
        abs2prom = self._var_allprocs_abs2prom['input']

        # inproms is a dict where the keys are promoted input names, and the values
        # are all of the absolute input names with that promoted name, which indicates
        # that the inputs will be connected to a common auto_ivc source in a future version.
        inproms = defaultdict(list)
        for n in self._var_allprocs_abs_names['input']:
            if n not in conns:
                inproms[abs2prom[n]].append(n)

        all_abs2meta = self._var_allprocs_abs2meta
        abs2meta = self._var_abs2meta

        for prom, tgts in inproms.items():
            if len(tgts) > 1:
                if prom in self._group_inputs:
                    gmeta = self._group_inputs[prom]
                else:
                    gmeta = ()

                tgt0 = tgts[0]
                if tgt0 in abs2meta:  # var is local
                    t0meta = abs2meta[tgt0]
                else:
                    t0meta = all_abs2meta[tgt0]
                t0units = t0meta['units']
                t0val = self._get_val(tgt0, kind='input', get_remote=True)

                for tgt in tgts[1:]:

                    if tgt in abs2meta:  # var is local
                        tmeta = abs2meta[tgt]
                    else:
                        tmeta = all_abs2meta[tgt]

                    tunits = tmeta['units'] if 'units' in tmeta else None
                    tval = self._get_val(tgt, kind='input', get_remote=True)

                    errs = []
                    if _has_val_mismatch(tunits, tval, t0units, t0val):
                        if 'units' not in gmeta and t0units != tunits:
                            errs.append('units')
                        if 'value' not in gmeta:
                            errs.append('value')

                    if errs:
                        inputs = list(sorted(tgts))
                        grpname = common_subpath(inputs)
                        simple_warning(f"{self.msginfo}: The following inputs, {inputs} are "
                                       f"connected but the metadata entries {errs} differ and "
                                       "have not been specified by Group.add_input.  This warning "
                                       "will become an error in a future version.  To remove the "
                                       f"abiguity, call {grpname}.add_input() and specify the "
                                       f"{errs} arg(s).")


    def _setup_connections(self):
        """
        Compute dict of all connections owned by this Group.
        """
        abs_in2out = self._conn_abs_in2out = {}
        global_abs_in2out = self._conn_global_abs_in2out
        pathname = self.pathname
        allprocs_discrete_in = self._var_allprocs_discrete['input']
        allprocs_discrete_out = self._var_allprocs_discrete['output']

        # Recursion
        for subsys in self._subsystems_myproc:
            subsys._setup_connections()

        if MPI:
            # collect set of local (not remote, not distributed) subsystems so we can
            # identify cross-process connections, which require the use of distributed
            # instead of purely local vector and transfer objects.
            self._local_system_set = set()
            for s in self._subsystems_myproc:
                if isinstance(s, Group):
                    self._local_system_set.update(s._local_system_set)
                elif not s.options['distributed']:
                    self._local_system_set.add(s.pathname)

        path_dot = pathname + '.' if pathname else ''
        path_len = len(path_dot)

        allprocs_abs2meta = self._var_allprocs_abs2meta

        nproc = self.comm.size

        # Check input/output units here, and set _has_input_scaling
        # to True for this Group if units are defined and different, or if
        # ref or ref0 are defined for the output.
        for abs_in, abs_out in global_abs_in2out.items():

            # First, check that this system owns both the input and output.
            if abs_in[:path_len] == path_dot and abs_out[:path_len] == path_dot:
                # Second, check that they are in different subsystems of this system.
                out_subsys = abs_out[path_len:].split('.', 1)[0]
                in_subsys = abs_in[path_len:].split('.', 1)[0]
                if out_subsys != in_subsys:
                    if abs_in in allprocs_discrete_in:
                        self._conn_discrete_in2out[abs_in] = abs_out
                    elif abs_out in allprocs_discrete_out:
                        msg = f"{self.msginfo}: Can't connect discrete output '{abs_out}' " + \
                              f"to continuous input '{abs_in}'."
                        if self._raise_connection_errors:
                            raise RuntimeError(msg)
                        else:
                            simple_warning(msg)
                    else:
                        abs_in2out[abs_in] = abs_out

                    if nproc > 1 and self._vector_class is None:
                        # check for any cross-process data transfer.  If found, use
                        # self._distributed_vector_class as our vector class.
                        in_path = abs_in.rsplit('.', 1)[0]
                        if in_path not in self._local_system_set:
                            self._vector_class = self._distributed_vector_class
                        else:
                            out_path = abs_out.rsplit('.', 1)[0]
                            if out_path not in self._local_system_set:
                                self._vector_class = self._distributed_vector_class

            # if connected output has scaling then we need input scaling
            if not self._has_input_scaling and not (abs_in in allprocs_discrete_in or
                                                    abs_out in allprocs_discrete_out):
                out_units = allprocs_abs2meta[abs_out]['units']
                in_units = allprocs_abs2meta[abs_in]['units']

                # if units are defined and different, we need input scaling.
                needs_input_scaling = (in_units and out_units and in_units != out_units)

                # we also need it if a connected output has any scaling.
                if not needs_input_scaling:
                    out_meta = allprocs_abs2meta[abs_out]

                    ref = out_meta['ref']
                    if np.isscalar(ref):
                        needs_input_scaling = ref != 1.0
                    else:
                        needs_input_scaling = np.any(ref != 1.0)

                    if not needs_input_scaling:
                        ref0 = out_meta['ref0']
                        if np.isscalar(ref0):
                            needs_input_scaling = ref0 != 0.0
                        else:
                            needs_input_scaling = np.any(ref0)

                        if not needs_input_scaling:
                            res_ref = out_meta['res_ref']
                            if np.isscalar(res_ref):
                                needs_input_scaling = res_ref != 1.0
                            else:
                                needs_input_scaling = np.any(res_ref != 1.0)

                self._has_input_scaling = needs_input_scaling

        # check compatability for any discrete connections
        for abs_in, abs_out in self._conn_discrete_in2out.items():
            in_type = self._var_allprocs_discrete['input'][abs_in]['type']
            try:
                out_type = self._var_allprocs_discrete['output'][abs_out]['type']
            except KeyError:
                msg = f"{self.msginfo}: Can't connect continuous output '{abs_out}' " + \
                      f"to discrete input '{abs_in}'."
                if self._raise_connection_errors:
                    raise RuntimeError(msg)
                else:
                    simple_warning(msg)
            if not issubclass(in_type, out_type):
                msg = f"{self.msginfo}: Type '{out_type.__name__}' of output '{abs_out}' is " + \
                      f"incompatible with type '{in_type.__name__}' of input '{abs_in}'."
                if self._raise_connection_errors:
                    raise RuntimeError(msg)
                else:
                    simple_warning(msg)

        # check unit/shape compatibility, but only for connections that are
        # either owned by (implicit) or declared by (explicit) this Group.
        # This way, we don't repeat the error checking in multiple groups.
        abs2meta = self._var_abs2meta

        for abs_in, abs_out in abs_in2out.items():
            # check unit compatibility
            out_units = allprocs_abs2meta[abs_out]['units']
            in_units = allprocs_abs2meta[abs_in]['units']

            if out_units:
                if not in_units:
                    msg = f"{self.msginfo}: Output '{abs_out}' with units of '{out_units}' " + \
                          f"is connected to input '{abs_in}' which has no units."
                    simple_warning(msg)
                elif not is_compatible(in_units, out_units):
                    msg = f"{self.msginfo}: Output units of '{out_units}' for '{abs_out}' " + \
                          f"are incompatible with input units of '{in_units}' for '{abs_in}'."
                    if self._raise_connection_errors:
                        raise RuntimeError(msg)
                    else:
                        simple_warning(msg)
            elif in_units is not None:
                msg = f"{self.msginfo}: Input '{abs_in}' with units of '{in_units}' is " + \
                      f"connected to output '{abs_out}' which has no units."
                simple_warning(msg)

            # check shape compatibility
            if abs_in in abs2meta and abs_out in abs2meta:



                # get output shape from allprocs meta dict, since it may
                # be distributed (we want global shape)
                out_shape = allprocs_abs2meta[abs_out]['global_shape']
                # get input shape and src_indices from the local meta dict
                # (input is always local)
                in_shape = abs2meta[abs_in]['shape']
                src_indices = abs2meta[abs_in]['src_indices']
                flat = abs2meta[abs_in]['flat_src_indices']

                if src_indices is None and out_shape != in_shape:
                    # out_shape != in_shape is allowed if
                    # there's no ambiguity in storage order
                    if not array_connection_compatible(in_shape, out_shape):
                        msg = f"{self.msginfo}: The source and target shapes do not match or " + \
                              f"are ambiguous for the connection '{abs_out}' to '{abs_in}'. " + \
                              f"The source shape is {tuple([int(s) for s in out_shape])} " + \
                              f"but the target shape is {tuple([int(s) for s in in_shape])}."
                        if self._raise_connection_errors:
                            raise ValueError(msg)
                        else:
                            simple_warning(msg)

                elif src_indices is not None:
                    src_indices = np.atleast_1d(src_indices)

                    if np.prod(src_indices.shape) == 0:
                        continue

                    # initial dimensions of indices shape must be same shape as target
                    for idx_d, inp_d in zip(src_indices.shape, in_shape):
                        if idx_d != inp_d:
                            msg = f"{self.msginfo}: The source indices " + \
                                  f"{src_indices} do not specify a " + \
                                  f"valid shape for the connection '{abs_out}' to " + \
                                  f"'{abs_in}'. The target shape is " + \
                                  f"{in_shape} but indices are {src_indices.shape}."
                            if self._raise_connection_errors:
                                raise ValueError(msg)
                            else:
                                simple_warning(msg)
                                continue

                    # any remaining dimension of indices must match shape of source
                    if len(src_indices.shape) > len(in_shape):
                        source_dimensions = src_indices.shape[len(in_shape)]
                        if source_dimensions != len(out_shape):
                            str_indices = str(src_indices).replace('\n', '')
                            msg = f"{self.msginfo}: The source indices " + \
                                  f"{str_indices} do not specify a " + \
                                  f"valid shape for the connection '{abs_out}' to '{abs_in}'. " + \
                                  f"The source has {len(out_shape)} dimensions but the " + \
                                  f"indices expect {source_dimensions}."
                            if self._raise_connection_errors:
                                raise ValueError(msg)
                            else:
                                simple_warning(msg)
                                continue
                    else:
                        source_dimensions = 1

                    # check all indices are in range of the source dimensions
                    if flat:
                        out_size = np.prod(out_shape)
                        mx = np.max(src_indices)
                        mn = np.min(src_indices)
                        if mx >= out_size:
                            bad_idx = mx
                        elif mn < -out_size:
                            bad_idx = mn
                        else:
                            bad_idx = None
                        if bad_idx is not None:
                            msg = f"{self.msginfo}: The source indices do not specify " + \
                                  f"a valid index for the connection '{abs_out}' to " + \
                                  f"'{abs_in}'. Index '{bad_idx}' is out of range for " + \
                                  f"a flat source of size {out_size}."
                            if self._raise_connection_errors:
                                raise ValueError(msg)
                            else:
                                simple_warning(msg)
                        if src_indices.ndim > 1:
                            abs2meta[abs_in]['src_indices'] = \
                                abs2meta[abs_in]['src_indices'].ravel()
                    else:
                        # For 1D source, we allow user to specify a flat list without setting
                        # flat_src_indices to True.
                        if src_indices.ndim == 1:
                            src_indices = src_indices[:, np.newaxis]

                        for d in range(source_dimensions):
                            if allprocs_abs2meta[abs_out]['distributed'] is True or \
                               allprocs_abs2meta[abs_in]['distributed'] is True:
                                d_size = out_shape[d] * self.comm.size
                            else:
                                d_size = out_shape[d]
                            arr = src_indices[..., d]
                            if np.any(arr >= d_size) or np.any(arr <= -d_size):
                                for i in arr.flat:
                                    if abs(i) >= d_size:
                                        msg = f"{self.msginfo}: The source indices " + \
                                              f"do not specify a valid index for the " + \
                                              f"connection '{abs_out}' to '{abs_in}'. " + \
                                              f"Index '{i}' is out of range for source " + \
                                              f"dimension of size {d_size}."
                                        if self._raise_connection_errors:
                                            raise ValueError(msg)
                                        else:
                                            simple_warning(msg)
