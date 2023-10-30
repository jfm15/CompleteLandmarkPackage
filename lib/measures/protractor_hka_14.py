import numpy as np
import torch

def l_angle(points):
        '''Claculates left and right angles of HKA, assuming the layout of txt points is 
        # f   centre of the femoral head
        # k   centre of the knee 
        # a   centre of the ankle
        # all distances are used squared as late as possible to avoid multiple sqrt
        '''
        coord_l = points[7:] 
        coord_ls = [coord_l]
        for coords in coord_ls:
            if (coords).dtype==str:
                f_coords=int(coords[0].split(',')[0]),int(coords[0].split(',')[1])
                k_coords=int(coords[2].split(',')[0]),int(coords[2].split(',')[1])
                a_coords=int(coords[5].split(',')[0]),int(coords[5].split(',')[1])
            else:
                points = points.detach().cpu().numpy()
                #print(points)
                f_coords=int(coords[0][0]),int(coords[0][1])
                k_coords=int(coords[2][0]),int(coords[2][1])
                a_coords=int(coords[5][0]),int(coords[5][1])

            fk=tuple(np.subtract(k_coords , f_coords))
            ka=tuple(np.subtract(a_coords , k_coords))
            
            dot = fk[0] * ka[0] + fk[1] * ka[1]
            fk_lengthsq = fk[0]**2 + fk[1]**2
            ka_lengthsq = ka[0]**2 + ka[1]**2
            
            angle = np.sqrt ((dot/fk_lengthsq) * (dot/ka_lengthsq))
            angle = round(np.degrees(np.arccos(angle)), 2)
            
            # determine whether k, f and a are in clockwise order
            # using the left predicate (area of parallelogram)
            varus_array = np.array([
                [k_coords[0], f_coords[0], a_coords[0]],
                [k_coords[1], f_coords[1], a_coords[1]],
                [1,1,1]
                ])
            varus = np.sign(np.linalg.det(varus_array))

            l_angle = varus * angle
            l_f_len=round(np.sqrt(fk_lengthsq),2)
            l_t_len=round(np.sqrt(ka_lengthsq),2)


            print("HKA left",angle,"degrees , femur length",round(np.sqrt(fk_lengthsq),2),"and tibia length",round(np.sqrt(ka_lengthsq),2))
            return l_angle
        
def r_angle(points):
    '''Claculates left and right angles of HKA, assuming the layout of txt points is 
    # f   centre of the femoral head
    # k   centre of the knee 
    # a   centre of the ankle
    # all distances are used squared as late as possible to avoid multiple sqrt
    '''
    coord_r = points[:7]
    coord_ls = [coord_r]
    for coords in coord_ls:
        if (coords).dtype==str:
            f_coords=int(coords[0].split(',')[0]),int(coords[0].split(',')[1])
            k_coords=int(coords[2].split(',')[0]),int(coords[2].split(',')[1])
            a_coords=int(coords[5].split(',')[0]),int(coords[5].split(',')[1])
        else:
            points = points.detach().cpu().numpy()
            f_coords=int(coords[0][0]),int(coords[0][1])
            k_coords=int(coords[2][0]),int(coords[2][1])
            a_coords=int(coords[5][0]),int(coords[5][1])

        fk=tuple(np.subtract(k_coords , f_coords))
        ka=tuple(np.subtract(a_coords , k_coords))
        
        dot = fk[0] * ka[0] + fk[1] * ka[1]
        fk_lengthsq = fk[0]**2 + fk[1]**2
        ka_lengthsq = ka[0]**2 + ka[1]**2
        
        angle = np.sqrt ((dot/fk_lengthsq) * (dot/ka_lengthsq))
        angle = round(np.degrees(np.arccos(angle)), 2)
        
        # determine whether k, f and a are in clockwise order
        # using the left predicate (area of parallelogram)
        varus_array = np.array([
            [k_coords[0], f_coords[0], a_coords[0]],
            [k_coords[1], f_coords[1], a_coords[1]],
            [1,1,1]
            ])
        varus = np.sign(np.linalg.det(varus_array))

        r_angle = varus * angle
        r_f_len=round(np.sqrt(fk_lengthsq),2)
        r_t_len=round(np.sqrt(ka_lengthsq),2)

        print("HKA right",angle,"degrees , femur length",round(np.sqrt(fk_lengthsq),2)," and tibia length",round(np.sqrt(ka_lengthsq),2))
        return r_angle
    