

class object2id(object):
    """
    Constructing dict for converting word to id.
    Params:
        def add_obj_id():
            name: 'char' or entity label
            obj: vocabulary path or entity label list
        
        def obj_2_id():
            name: 'char'
            obj: a phrase or all phrases concated
    Return:
        function obj_2_id() return list of index for embedding
    """
    def __init__(self):
        self.diction_dict={}

    def add_obj_id(self, name, obj):
        name = name.upper()
        if isinstance(obj, list):
            self.diction_dict[name] = obj
        elif isinstance(obj, str):
            if name.upper() == 'CHAR':
                content = open(obj, 'r', encoding='utf8').read().splitlines()
                self.diction_dict[name] = []
                for i, c in enumerate(content):
                    if c == '_换行符_':
                        c = '\n'
                    self.diction_dict[name].append(c)  
            else:
                self.diction_dict[name] = eval(open(obj, 'r', encoding='utf8').read())

    def obj_2_id(self, name, obj):
        name = name.upper()
        try:
            if name.upper() == 'CHAR':
                if isinstance(obj, str):
                    obj = list(obj)
                if isinstance(obj, list):
                    return list(map(self.char_2_id, obj))
            else:
                return self.diction_dict[name].index(obj)
        except:
            print(name, obj)
            return 0
            # raise RuntimeError('Error')

    def char_2_id(self, char):
        if char not in self.diction_dict['CHAR']:
            return self.diction_dict['CHAR'].index('[UNK]')
        else:
            return self.diction_dict['CHAR'].index(char)

    def char_get_id(self, char):
        return self.char_2_id(char)

    def get_convertion_list(self, name):
        name = name.upper()
        return self.diction_dict[name]

    @property
    def info(self):
        for k, v in self.diction_dict.items():
            print('{}有{}维'.format(k,len(v)))

    def id_2_obj(self, name, id_):
        name = name.upper()
        if isinstance(id_, list):
            return list(map(lambda x:self.diction_dict[name][x], id_))
        return self.diction_dict[name][id_]

    def get_len(self, name):
        name = name.upper()
        if name in self.diction_dict.keys():
            return len(self.diction_dict[name])
        else:
            return -1

    def get_label(self, name):
        name = name.upper()
        if name in self.diction_dict.keys():
            print(self.diction_dict[name])  