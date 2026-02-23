# Norma para mantener variables SNOMED

Se ha realizado un cribado de las variables SNOMED CT para mantener solo aquellas que tengan un porcentaje de no nulos específico según las categoria SNOMED a la que pertenece.

Todas las columnas completamente nulas se han eliminado. Tras esto, según la categoría:

- disorder:Se mantienen todas tal cual en este primer cribado. Se aplicará un intento de agrupación mediante sistema jerarquico de SNOMED.
- procedure: Se eliminan todas las que tienen un porcentaje de no nulos menor al 0.1% y se mantienen el resto para la agrupación
- finding: Se eliminan todas las que tienen un porcentaje de no nulos menor al 0.1% y se mantienen el resto para la agrupación.
- regime/therapy: Se mantienen todas.
- Observable entity: Se mantienen todas.
- Resto: Se eliminan la mayoría menos alguna suelta con relevancia clínica.