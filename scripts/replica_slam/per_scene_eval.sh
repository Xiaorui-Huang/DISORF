# offline
# bash scripts/replica_slam/splatfacto_eval.sh tx2_office0 69
# bash scripts/replica_slam/splatfacto_eval.sh tx2_office1 73
# bash scripts/replica_slam/splatfacto_eval.sh tx2_office2 60
# bash scripts/replica_slam/splatfacto_eval.sh tx2_office3 64
# bash scripts/replica_slam/splatfacto_eval.sh tx2_office4 67
# bash scripts/replica_slam/splatfacto_eval.sh tx2_room0 59
# bash scripts/replica_slam/splatfacto_eval.sh tx2_room1 59
# bash scripts/replica_slam/splatfacto_eval.sh tx2_room2 60

# online
# for i in {1..3}; do
#     for sampl in inverse_delta; do
#         # bash scripts/replica_slam/splatfacto_eval_online.sh tx2_office0 69 SO3xR3 $sampl _v2
#         bash scripts/replica_slam/splatfacto_eval_online.sh tx2_office1 73 SO3xR3 $sampl _v2
#         bash scripts/replica_slam/splatfacto_eval_online.sh tx2_office2 60 SO3xR3 $sampl _v2
#         bash scripts/replica_slam/splatfacto_eval_online.sh tx2_office3 64 SO3xR3 $sampl _v2
#         bash scripts/replica_slam/splatfacto_eval_online.sh tx2_office4 67 SO3xR3 $sampl _v2
#         bash scripts/replica_slam/splatfacto_eval_online.sh tx2_room0 59 SO3xR3 $sampl _v2
#         bash scripts/replica_slam/splatfacto_eval_online.sh tx2_room1 59 SO3xR3 $sampl _v2
#         bash scripts/replica_slam/splatfacto_eval_online.sh tx2_room2 60 SO3xR3 $sampl _v2
#     done
# done

# for i in {1..2}; do
#     for sampl in inverse_delta; do
#         bash scripts/replica_slam/splatfacto_eval_online.sh tx2_office0 69 SO3xR3 $sampl _s2_R3 2 3
#         bash scripts/replica_slam/splatfacto_eval_online.sh tx2_office1 73 SO3xR3 $sampl _s2_R3 2 3
#         bash scripts/replica_slam/splatfacto_eval_online.sh tx2_office2 60 SO3xR3 $sampl _s2_R3 2 3
#         bash scripts/replica_slam/splatfacto_eval_online.sh tx2_office3 64 SO3xR3 $sampl _s2_R3 2 3
#         bash scripts/replica_slam/splatfacto_eval_online.sh tx2_office4 67 SO3xR3 $sampl _s2_R3 2 3
#         bash scripts/replica_slam/splatfacto_eval_online.sh tx2_room0 59 SO3xR3 $sampl _s2_R3 2 3
#         bash scripts/replica_slam/splatfacto_eval_online.sh tx2_room1 59 SO3xR3 $sampl _s2_R3 2 3
#         bash scripts/replica_slam/splatfacto_eval_online.sh tx2_room2 60 SO3xR3 $sampl _s2_R3 2 3
#     done
# done

# for i in {1..2}; do
#     for sampl in inverse_delta; do
#         bash scripts/replica_slam/splatfacto_eval_online.sh tx2_office0 69 SO3xR3 $sampl _s1_R5 1 5
#         bash scripts/replica_slam/splatfacto_eval_online.sh tx2_office1 73 SO3xR3 $sampl _s1_R5 1 5
#         bash scripts/replica_slam/splatfacto_eval_online.sh tx2_office2 60 SO3xR3 $sampl _s1_R5 1 5
#         bash scripts/replica_slam/splatfacto_eval_online.sh tx2_office3 64 SO3xR3 $sampl _s1_R5 1 5
#         bash scripts/replica_slam/splatfacto_eval_online.sh tx2_office4 67 SO3xR3 $sampl _s1_R5 1 5
#         bash scripts/replica_slam/splatfacto_eval_online.sh tx2_room0 59 SO3xR3 $sampl _s1_R5 1 5
#         bash scripts/replica_slam/splatfacto_eval_online.sh tx2_room1 59 SO3xR3 $sampl _s1_R5 1 5
#         bash scripts/replica_slam/splatfacto_eval_online.sh tx2_room2 60 SO3xR3 $sampl _s1_R5 1 5
#     done
# done

# for i in {1..2}; do
#     for sampl in default imap inverse_delta; do
#         bash scripts/replica_slam/splatfacto_eval_online_mvs.sh tx2_office0 69 SO3xR3 $sampl _mvs
#         bash scripts/replica_slam/splatfacto_eval_online_mvs.sh tx2_office1 73 SO3xR3 $sampl _mvs
#         bash scripts/replica_slam/splatfacto_eval_online_mvs.sh tx2_office2 60 SO3xR3 $sampl _mvs
#         bash scripts/replica_slam/splatfacto_eval_online_mvs.sh tx2_office3 64 SO3xR3 $sampl _mvs
#         bash scripts/replica_slam/splatfacto_eval_online_mvs.sh tx2_office4 67 SO3xR3 $sampl _mvs
#         bash scripts/replica_slam/splatfacto_eval_online_mvs.sh tx2_room0 59 SO3xR3 $sampl _mvs
#         bash scripts/replica_slam/splatfacto_eval_online_mvs.sh tx2_room1 59 SO3xR3 $sampl _mvs
#         bash scripts/replica_slam/splatfacto_eval_online_mvs.sh tx2_room2 60 SO3xR3 $sampl _mvs
#     done
# done


# for i in {1..2}; do
#     for sampl in hierarchical_loss; do
#         bash scripts/replica_slam/splatfacto_eval_online.sh tx2_office0 69 SO3xR3 $sampl
#         bash scripts/replica_slam/splatfacto_eval_online.sh tx2_office1 73 SO3xR3 $sampl
#         bash scripts/replica_slam/splatfacto_eval_online.sh tx2_office2 60 SO3xR3 $sampl
#         bash scripts/replica_slam/splatfacto_eval_online.sh tx2_office3 64 SO3xR3 $sampl
#         bash scripts/replica_slam/splatfacto_eval_online.sh tx2_office4 67 SO3xR3 $sampl
#         bash scripts/replica_slam/splatfacto_eval_online.sh tx2_room0 59 SO3xR3 $sampl
#         bash scripts/replica_slam/splatfacto_eval_online.sh tx2_room1 59 SO3xR3 $sampl
#         bash scripts/replica_slam/splatfacto_eval_online.sh tx2_room2 60 SO3xR3 $sampl
#     done
# done

for i in {1..2}; do
    for sampl in hybrid; do
        bash scripts/replica_slam/splatfacto_eval_online_mvs.sh tx2_office0 69 SO3xR3 $sampl _mvs
        bash scripts/replica_slam/splatfacto_eval_online_mvs.sh tx2_office1 73 SO3xR3 $sampl _mvs
        bash scripts/replica_slam/splatfacto_eval_online_mvs.sh tx2_office2 60 SO3xR3 $sampl _mvs
        bash scripts/replica_slam/splatfacto_eval_online_mvs.sh tx2_office3 64 SO3xR3 $sampl _mvs
        bash scripts/replica_slam/splatfacto_eval_online_mvs.sh tx2_office4 67 SO3xR3 $sampl _mvs
        bash scripts/replica_slam/splatfacto_eval_online_mvs.sh tx2_room0 59 SO3xR3 $sampl _mvs
        bash scripts/replica_slam/splatfacto_eval_online_mvs.sh tx2_room1 59 SO3xR3 $sampl _mvs
        bash scripts/replica_slam/splatfacto_eval_online_mvs.sh tx2_room2 60 SO3xR3 $sampl _mvs
    done
done