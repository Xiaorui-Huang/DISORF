POSE=$1

for sampl in default imap inverse_delta hybrid; do
    bash scripts/tnt/splatfacto_eval_online_mvs.sh Barn 532 ${POSE} $sampl _ucmvs uncontract 
    bash scripts/tnt/splatfacto_eval_online_mvs.sh Train 484 ${POSE} $sampl _ucmvs uncontract
    bash scripts/tnt/splatfacto_eval_online_mvs.sh Truck 314 ${POSE} $sampl _ucmvs uncontract
done

for i in {1..2}; do
    for sampl in inverse_delta hybrid; do
        bash scripts/tnt/splatfacto_eval_online_mvs.sh Barn 532 ${POSE} $sampl _ucmvs uncontract 
        bash scripts/tnt/splatfacto_eval_online_mvs.sh Train 484 ${POSE} $sampl _ucmvs uncontract 
        bash scripts/tnt/splatfacto_eval_online_mvs.sh Truck 314 ${POSE} $sampl _ucmvs uncontract 
    done
done
