# Norma para mantener variables SNOMED

Se ha realizado un cribado de las variables SNOMED CT para mantener solo aquellas que tengan un porcentaje de no nulos específico según las categoria SNOMED a la que pertenece.

Todas las columnas completamente nulas se han eliminado. Tras esto, según la categoría:

- disorder:Se mantienen todas tal cual en este primer cribado. Se aplicará un intento de agrupación mediante sistema jerarquico de SNOMED.
- procedure: Se eliminan todas las que tienen un porcentaje de no nulos menor al 0.1% y se mantienen el resto para la agrupación. Se eliminan procedimientos que se solicitan cuando ya existe alta sospecha de cáncer de pulmón. (C-367336001) (C-444260001) (C-86964003)

- finding: Se eliminan todas las que tienen un porcentaje de no nulos menor al 0.1% y se mantienen el resto para la agrupación.
- regime/therapy: Se mantienen todas.
- Observable entity: Se mantienen todas.
- Resto: Se eliminan la mayoría menos alguna suelta con relevancia clínica.

# Norma para mantener medicamentos RxNorm

Se mapean a ATC nivel 3

# Norma para mantener LOINC.

Se eliminan aquellos pocos convencionales o que son solicitados unicamente cuando ya hay un cáncer previamente diagnosticado.

C-85318-4 / C-434363004
C-85337-4 / C-85339-0 / C-10480-2
C-21924-6
 (C-88040-1)