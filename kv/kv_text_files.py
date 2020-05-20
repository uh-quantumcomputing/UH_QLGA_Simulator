run = ['''
<PhysicsModeling>:
    name: "physicsmodel"
#    canvas.before:
#        Color:
#            rgb: .2, .2, .2
#        Rectangle:
#            size: self.size
#            source: '/background.png'

    # this below is variables in the .py file being assigned to the id's
    # below in the .kv file here to allow them to be referenced in the core code
    # format below is "[.py variable name]: [id name in .kv file]"
    physicsModel: physicsModel
    initialConditions: initialConditions
    visTechnique: visTechnique

    FloatLayout:
        Label:
            text: "[size=35dp][b]QLGA SIMULATOR[/b][/size] [size=20dp][b]Physics Modeling[/b][/size]"
            markup: True
            size_hint: (0.75, 0.5)
            pos_hint: {'center': (0.5, 0.75)}
            halign: 'center'


        Label:
            text: "Model"
            halign: 'center'
            pos_hint: {'center': (0.15, .5)}

        Spinner:
            id: physicsModel
            size_hint: (0.3, 0.1)
            pos_hint: {'center': (0.35, .5)}
            text: 'Click to Select'
            halign: 'center'
            values: ''',
            '''on_text: root.physicsmodel_spinner_clicked(physicsModel.text)

        Label:
            text: "Initial Conditions "
            size_hint: (0.75, 0.5)
            pos_hint: {'center': (0.6, 0.50)}

        Spinner:
            id: initialConditions
            size_hint: (0.2, 0.1)
            pos_hint: {'center': (0.8, .5)}
            text: 'Click to Select'
            halign: 'center'
            values:''', 
            '''on_text: root.initialconditions_spinner_clicked(initialConditions.text)



        Label:
            text: "Visualization"
            halign: 'center'
            pos_hint: {'center': (0.37, .3)}

        Spinner:
            id: visTechnique
            size_hint: (0.24, 0.1)
            pos_hint: {'center': (0.57, .3)}
            text: 'Click to Select'
            halign: 'center'
            values:''',
            '''on_text: root.visualization_spinner_clicked(visTechnique.text)

        Button:
            text: '[size=15dp][b]<-[/b][/size] Go Back'
            markup: True
            size_hint: (0.15, 0.05)
            pos_hint: {'center': (0.15, 0.07)}
            on_release:
                app.root.current = "runoptions"
                root.manager.transition.direction = "right"

        Button:
            text: 'Next Page [size=15dp][b]->[/b][/size]'
            markup: True
            halign: 'center'
            size_hint: (0.15, 0.05)
            pos_hint: {'center': (0.85, 0.07)}
            on_press: root.next_button()
            on_release:
                app.root.current = "kwargOptions"
                root.manager.transition.direction = "left"
'''
]