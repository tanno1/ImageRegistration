# image registration
# match corners

import cv2

def match(img_gray, ref_gray, unreg_read_img, ref_read_img):
    orb = cv2.ORB_create()
    kp_img, desc_img = orb.detectAndCompute(img_gray, None)
    kp_ref, desc_ref = orb.detectAndCompute(ref_gray, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    matches = bf.match(desc_img, desc_ref)
    matches = sorted(matches, key=lambda x: x.distance)

    # keep good points, basically a threshold
    # good_matches = []
    # for m in matches:
    #     if m.distance <0.7 * matches[-1].distance:
    #         good_matches.append(m)
    
    print('matches:')
    print(matches)
    # draw match result
    match_image = cv2.drawMatches(unreg_read_img, kp_img, ref_read_img, kp_ref, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    print(match_image)
    print(type(match_image))

    return match_image

if __name__ == '__main__':
    match()